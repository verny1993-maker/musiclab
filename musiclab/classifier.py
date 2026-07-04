"""
musiclab.classifier — Genre classifier from 8D audio vectors.

PyTorch neural network that classifies tracks into genres using the 8D
embedding vectors (BPM, key cos/sin, energy, danceability, MFCC×3).

Architecture: 8 → 64 → 32 → N_genres (3-layer MLP with dropout).

Usage:
    from musiclab.classifier import GenreClassifier, train, predict

    model = GenreClassifier(num_genres=10)
    metrics = train(model, train_loader, val_loader, epochs=50)
    genre = predict(model, vector=[0.5, 0.0, 1.0, 0.7, 0.65, 0.4, 0.5, 0.6])
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════


class GenreClassifier(nn.Module):
    """3-layer MLP for genre classification from 8D audio vectors."""

    def __init__(self, input_dim: int = 8, hidden: int = 64, num_genres: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_genres),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════


def _load_data(
    tracks_path: str | Path,
    db_path: str | Path,
    top_n_genres: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Load 8D vectors + genre labels from JSON and SQLite.

    Returns:
        X: tensor of shape (N, 8) — normalized vectors
        y: tensor of shape (N,) — genre class indices
        genre_names: list of genre names (index → name)
    """
    tracks_path = Path(tracks_path)
    db_path = Path(db_path)

    # Load tracks with genres
    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)

    # Load vectors from SQLite
    conn = sqlite3.connect(str(db_path))
    vectors_db = {}
    for row in conn.execute(
        "SELECT track_id, vector_8d FROM audio_analysis WHERE vector_8d IS NOT NULL"
    ):
        vectors_db[row[0]] = json.loads(row[1])
    conn.close()

    # Match tracks with vectors + genres, pick top genre per track
    all_genres: list[str] = []
    matched = []
    for t in tracks:
        tid = t.get("id")
        genres = (t.get("meta") or {}).get("genres") or []
        if not genres or tid not in vectors_db:
            continue
        top_genre = genres[0]  # first genre is most specific/confident
        all_genres.append(top_genre)
        matched.append((vectors_db[tid], top_genre))

    # Select top-N genres, rest → "other"
    genre_counts = Counter(all_genres)
    top_genres = [g for g, _ in genre_counts.most_common(top_n_genres)]
    genre_to_idx = {g: i for i, g in enumerate(top_genres)}

    X_list, y_list = [], []
    for vec, genre in matched:
        if genre in genre_to_idx:
            X_list.append(vec)
            y_list.append(genre_to_idx[genre])

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    return X, y, top_genres


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════


def train(
    model: GenreClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cpu",
    class_weights: torch.Tensor | None = None,
) -> dict:
    """
    Train the genre classifier.

    Returns:
        dict with keys: train_losses, val_losses, val_accuracies, best_epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / total
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch + 1

        scheduler.step()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
    }


def evaluate(
    model: GenreClassifier,
    loader: DataLoader,
    genre_names: list[str],
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on a test set. Returns per-class metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    # Compute per-class accuracy
    correct_per_class = Counter()
    total_per_class = Counter()
    for pred, label in zip(all_preds, all_labels):
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1

    per_class = {}
    for idx, name in enumerate(genre_names):
        total = total_per_class.get(idx, 0)
        correct = correct_per_class.get(idx, 0)
        per_class[name] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "support": total,
        }

    overall = sum(correct_per_class.values()) / max(sum(total_per_class.values()), 1)

    return {"overall_accuracy": overall, "per_class": per_class}


def predict(
    model: GenreClassifier,
    vector: list[float],
    genre_names: list[str],
    device: str = "cpu",
) -> list[tuple[str, float]]:
    """Predict genre probabilities for a single 8D vector."""
    model.eval()
    x = torch.tensor([vector], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    ranked = sorted(
        [(genre_names[i], float(probs[i])) for i in range(len(genre_names))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked

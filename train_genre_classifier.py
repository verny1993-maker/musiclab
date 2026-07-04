#!/usr/bin/env python3
"""
train_genre_classifier.py — Train a genre classifier on 8D audio vectors.

Loads tracks_enriched_all.json + audio_analysis.db, builds a PyTorch MLP,
trains on top-N genres, evaluates, and saves the model.

Usage:
    python train_genre_classifier.py
    python train_genre_classifier.py --epochs 100 --genres 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from musiclab.classifier import GenreClassifier, _load_data, train


def main():
    parser = argparse.ArgumentParser(description="Train genre classifier")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--genres", type=int, default=8, help="Top-N genres")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--tracks",
        default=str(_PROJECT_ROOT / "data" / "library" / "tracks_enriched_all.json"),
    )
    parser.add_argument(
        "--db",
        default=str(_PROJECT_ROOT / "data" / "library" / "audio_analysis.db"),
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "models" / "genre_classifier.pt"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Top-N genres: {args.genres}")
    print(f"Epochs: {args.epochs}")

    # ── Load data ──
    print("\nLoading data...")
    X, y, genre_names = _load_data(
        tracks_path=args.tracks,
        db_path=args.db,
        top_n_genres=args.genres,
    )
    print(f"Matched {len(X)} tracks across {len(genre_names)} genres")
    print(f"Genres: {genre_names}")

    # Show class distribution
    class_counts = [
        (genre_names[i], (y == i).sum().item()) for i in range(len(genre_names))
    ]
    for name, count in class_counts:
        print(f"  {name}: {count}")

    if len(X) < 50:
        print("ERROR: Not enough data. Need at least 50 tracks with vectors + genres.")
        sys.exit(1)

    # ── Train/val/test split (70/15/15) ──
    n = len(X)
    indices = torch.randperm(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ── Model ──
    model = GenreClassifier(input_dim=8, hidden=64, num_genres=args.genres)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Compute class weights (inverse frequency)
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # ── Train ──
    print("\nTraining...")
    metrics = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        class_weights=class_weights,
    )

    print(
        f"\nBest validation accuracy: {metrics['best_accuracy']:.3f} (epoch {metrics['best_epoch']})"
    )
    print(f"Final train loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Final val loss: {metrics['val_losses'][-1]:.4f}")

    # ── Evaluate on test set ──
    print("\nTest set evaluation:")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=genre_names,
            digits=3,
        )
    )

    # ── Save model ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "genre_names": genre_names,
            "input_dim": 8,
            "hidden": 64,
            "num_genres": args.genres,
            "metrics": metrics,
        },
        output_path,
    )
    print(f"Model saved to {output_path}")

    # ── Sample predictions ──
    print("\nSample predictions (test set):")
    for i in range(min(5, len(X_test))):
        probs = torch.softmax(model(X_test[i : i + 1].to(device)), dim=1)[0]
        top3 = sorted(
            [(genre_names[j], float(probs[j])) for j in range(len(genre_names))],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        true_genre = genre_names[y_test[i].item()]
        pred_str = " | ".join(f"{g}: {p:.2f}" for g, p in top3)
        print(f"  True: {true_genre:20s} → {pred_str}")


if __name__ == "__main__":
    main()

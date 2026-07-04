# Deploying Music Analysis API

The Music Analysis API (`server.py`) is a FastAPI service that extracts musical features
from WAV files: BPM, key, Camelot, energy, danceability, MFCC, and beat grid.

## Architecture

```
                         AWS Lightsail
┌──────────┐    POST /analyze    ┌──────────────────┐
│  Client  │ ──────────────────→ │  Container (nano) │
│  (curl)  │ ←── JSON ────────── │  FastAPI :8777    │
└──────────┘                     │  1 vCPU / 512 MB  │
                                 └──────────────────┘
```

## Option A: AWS Lightsail (Terraform)

### Prerequisites

1. [AWS account](https://aws.amazon.com/free/) (free tier works)
2. [AWS CLI](https://aws.amazon.com/cli/) installed + configured:
   ```bash
   aws configure
   # Enter: Access Key ID, Secret Access Key, region (us-east-1)
   ```
3. [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5
4. [Docker](https://docs.docker.com/get-docker/)

### Step 1: Build and push Docker image

```bash
# From the repo root:
docker build -t musiclab-api -f deploy/Dockerfile .

# Push to Lightsail container registry
aws lightsail create-container-service --service-name musiclab-api --power nano --scale 1

aws lightsail push-container-image \
    --service-name musiclab-api \
    --label musiclab-api \
    --image musiclab-api:latest
```

### Step 2: Deploy with Terraform

```bash
cd deploy/terraform

# Get the image digest from Step 1
IMAGE=$(aws lightsail get-container-images \
    --service-name musiclab-api \
    --query 'containerImages[0].image' --output text)

terraform init
terraform plan -var="container_image=$IMAGE"
terraform apply -var="container_image=$IMAGE" -auto-approve
```

### Step 3: Test

```bash
API_URL=$(terraform output -raw api_url)

# Health check
curl "$API_URL/docs"

# Analyze a WAV file (requires file accessible inside container)
curl -X POST "$API_URL/analyze" \
    -H "Content-Type: application/json" \
    -d '{"filepath": "/tmp/analysis/test.wav"}'
```

### Cleanup

```bash
terraform destroy -auto-approve
```

## Option B: GCP Cloud Run (simpler, free tier)

```bash
# Enable services
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/musiclab-api -f deploy/Dockerfile .

# Deploy
gcloud run deploy musiclab-api \
    --image gcr.io/$PROJECT_ID/musiclab-api \
    --port 8777 \
    --memory 512Mi \
    --cpu 1 \
    --allow-unauthenticated \
    --region us-central1
```

## Option C: Local Docker (for development)

```bash
docker build -t musiclab-api -f deploy/Dockerfile .
docker run -d -p 8777:8777 \
    -v $(pwd)/data/tmp_analysis:/tmp/analysis \
    --name musiclab-api \
    musiclab-api

# Test
curl -X POST http://localhost:8777/analyze \
    -H "Content-Type: application/json" \
    -d '{"filepath": "/tmp/analysis/test.wav"}'
```

## API Reference

### POST /analyze

Analyze a WAV file for musical features.

**Request:**
```json
{
    "filepath": "/tmp/analysis/track.wav"
}
```

**Response:**
```json
{
    "filepath": "/tmp/analysis/track.wav",
    "duration": 234.5,
    "sample_rate": 22050,
    "bpm": 128.5,
    "key": "A",
    "camelot": "8A",
    "key_strength": 0.87,
    "energy": 0.72,
    "danceability": 0.65,
    "loudness": -8.3,
    "mfcc_mean": [-140.2, 85.3, -12.1],
    "spectral_centroid": 2100.5,
    "zero_crossing_rate": 0.08,
    "beat_positions": [0.0, 0.47, 0.94, ...]
}
```

## Cost

| Provider | Service | Monthly Cost |
|----------|---------|-------------|
| AWS Lightsail | nano (512MB) | ~$7 |
| GCP Cloud Run | 512MB, pay-per-request | ~$0-5 |
| Local Docker | Your machine | $0 |

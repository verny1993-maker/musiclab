# Music Analysis API — Terraform Deployment
#
# Deploys the FastAPI audio analysis service to AWS Lightsail Containers.
#
# Prerequisites:
#   1. AWS account (free tier works)
#   2. AWS CLI installed + configured: aws configure
#   3. Docker installed
#   4. Terraform installed: https://developer.hashicorp.com/terraform/downloads
#
# Usage:
#   cd deploy/terraform
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ═══════════════════════════════════════════════════════════
# Variables
# ═══════════════════════════════════════════════════════════

variable "aws_region" {
  description = "AWS region for Lightsail"
  type        = string
  default     = "us-east-1"
}

variable "service_name" {
  description = "Lightsail container service name"
  type        = string
  default     = "musiclab-api"
}

variable "container_image" {
  description = "Docker image tag (pushed to Lightsail)"
  type        = string
}

# ═══════════════════════════════════════════════════════════
# Lightsail Container Service
# ═══════════════════════════════════════════════════════════

resource "aws_lightsail_container_service" "api" {
  name        = var.service_name
  power       = "nano" # $7/month — smallest instance
  scale       = 1
  is_disabled = false

  # Public endpoint configuration
  public_domain_names {
    certificate {
      certificate_name = "${var.service_name}-cert"
    }
  }
}

resource "aws_lightsail_container_service_deployment_version" "api" {
  service_name = aws_lightsail_container_service.api.name

  container {
    container_name = "music-analysis"
    image          = var.container_image

    ports = {
      8777 = "HTTP"
    }

    environment = {
      PYTHONUNBUFFERED = "1"
    }
  }

  public_endpoint {
    container_name = "music-analysis"
    container_port = 8777

    health_check {
      healthy_threshold   = 2
      unhealthy_threshold = 2
      timeout_seconds     = 5
      interval_seconds    = 15
      path                = "/docs"
      success_codes       = "200"
    }
  }
}

# ═══════════════════════════════════════════════════════════
# Outputs
# ═══════════════════════════════════════════════════════════

output "api_url" {
  description = "Public API endpoint"
  value       = aws_lightsail_container_service.api.url
}

output "analyze_endpoint" {
  description = "POST endpoint for audio analysis"
  value       = "${aws_lightsail_container_service.api.url}/analyze"
}

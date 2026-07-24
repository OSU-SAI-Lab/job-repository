#!/bin/bash

# Deployment script for SAM3 service on NRP Nautilus
# This script deploys the SAM3 FastAPI service along with Redis using Kubernetes

set -e  # Exit on any error

if [ -z "$1" ]; then
	echo "Usage: $0 <HF_TOKEN>"
	echo "Example: $0 hf_xxxxxxxxxxxxxxxxx"
	exit 1
fi

HF_TOKEN="$1"

echo "Starting deployment of SAM3 service..."

# Create or update secret securely from argument (not committed in YAML)
kubectl create secret generic sam3-secrets \
	--from-literal=HF_TOKEN="$HF_TOKEN" \
	--dry-run=client -o yaml | kubectl apply -f -

# Apply the Kubernetes manifests
kubectl apply -f sam3_nrp_deployment.yaml

echo "Waiting for deployments to be ready..."

# Wait for Redis deployment
kubectl wait --for=condition=available --timeout=300s deployment/redis-server

# Wait for SAM3 deployment
kubectl wait --for=condition=available --timeout=600s deployment/sam3-fastapi

echo "Deployment completed successfully!"
echo "SAM3 service should be accessible at: https://sam3-sailab.nrp-nautilus.io"
echo ""
echo "To check the status of your deployments:"
echo "kubectl get pods"
echo "kubectl get services"
echo "kubectl get ingress"
#!/bin/bash

# MLOps Iris Classification - Deployment Script
# Usage: ./scripts/deploy.sh [environment] [version]

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "🚀 Deploying MLOps Iris to $ENVIRONMENT environment (version: $VERSION)"

# Validate environment
case $ENVIRONMENT in
    staging|production)
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT. Use 'staging' or 'production'"
        exit 1
        ;;
esac

# Set environment-specific variables
if [ "$ENVIRONMENT" = "staging" ]; then
    IMAGE_TAG="${VERSION}-staging"
    NAMESPACE="mlops-staging"
    REPLICAS=1
else
    IMAGE_TAG="$VERSION"
    NAMESPACE="mlops-production"
    REPLICAS=3
fi

IMAGE_NAME="ghcr.io/${GITHUB_REPOSITORY:-your-org/mlops-iris}"

echo "📦 Using image: $IMAGE_NAME:$IMAGE_TAG"
echo "🏗️  Namespace: $NAMESPACE"
echo "🔄 Replicas: $REPLICAS"

# Create namespace if it doesn't exist
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Deploy to Kubernetes
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-iris
  namespace: $NAMESPACE
  labels:
    app: mlops-iris
    environment: $ENVIRONMENT
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: mlops-iris
  template:
    metadata:
      labels:
        app: mlops-iris
        environment: $ENVIRONMENT
    spec:
      containers:
      - name: mlops-iris
        image: $IMAGE_NAME:$IMAGE_TAG
        ports:
        - containerPort: 80
        env:
        - name: MODEL_PATH
          value: "/app/artifacts/model.pkl"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mlops-iris-service
  namespace: $NAMESPACE
  labels:
    app: mlops-iris
    environment: $ENVIRONMENT
spec:
  selector:
    app: mlops-iris
  ports:
    - port: 80
      targetPort: 80
      protocol: TCP
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-iris-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: iris-$ENVIRONMENT.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlops-iris-service
            port:
              number: 80
EOF

# Wait for rollout to complete
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/mlops-iris -n "$NAMESPACE" --timeout=300s

# Run post-deployment tests
echo "🧪 Running post-deployment tests..."
sleep 10

# Get service URL
SERVICE_IP=$(kubectl get svc mlops-iris-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
echo "🌐 Service available at: $SERVICE_IP"

# Test health endpoint
if kubectl run test-health --image=curlimages/curl --rm -i --restart=Never -- curl -f "$SERVICE_IP/health"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo "🎉 Deployment to $ENVIRONMENT completed successfully!"
echo "📊 Check status: kubectl get pods -n $NAMESPACE"

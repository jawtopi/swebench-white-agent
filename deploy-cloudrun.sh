#!/bin/bash
# Deploy SWE-bench White Agent to Google Cloud Run
#
# Prerequisites:
# 1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
# 2. Authenticate: gcloud auth login
# 3. Set project: gcloud config set project YOUR_PROJECT_ID
# 4. Enable APIs: gcloud services enable run.googleapis.com artifactregistry.googleapis.com

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-swebench-white-agent}"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}"

# Validate required env vars
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is required"
    echo "Set it with: export OPENAI_API_KEY=your-key"
    exit 1
fi

echo "Deploying SWE-bench White Agent to Cloud Run"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Service: ${SERVICE_NAME}"
echo ""

# Build the container image using Dockerfile.cloudrun
echo "Building container image..."
gcloud builds submit --config cloudbuild.yaml --project "${PROJECT_ID}"

# Get the expected service host (without https://)
# Format: SERVICE_NAME-PROJECT_NUMBER.REGION.run.app
SERVICE_HOST="${SERVICE_NAME}-$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)').${REGION}.run.app"
SERVICE_URL="https://${SERVICE_HOST}"

# Deploy to Cloud Run with extended timeout
# Use same env vars as Railway: CLOUDRUN_HOST, HTTPS_ENABLED, etc.
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600 \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    --set-env-vars "PYTHONUNBUFFERED=1" \
    --set-env-vars "AGENT_URL=${SERVICE_URL}" \
    --set-env-vars "HOST=0.0.0.0" \
    --set-env-vars "REPOS_DIR=/tmp/swebench_repos"

echo ""
echo "Deployment complete!"
echo "  Service URL: ${SERVICE_URL}"
echo "  Agent Card: ${SERVICE_URL}/.well-known/agent.json"
echo ""
echo "To use with AgentBeats green agent, add this agent URL:"
echo "  ${SERVICE_URL}"
echo ""
echo "Timeout is set to 600 seconds (10 minutes) for long-running SWE-bench tasks."

#!/bin/bash
set -e

echo "=== LLAVA-LENS HuggingFace Spaces Deployment ==="
echo ""

REPO_NAME="${1:-llava-lens}"
HF_USERNAME="${2:-}"

if [ -z "$HF_USERNAME" ]; then
    echo "Usage: ./deploy.sh <repo-name> <hf-username>"
    echo "Example: ./deploy.sh llava-lens yourusername"
    exit 1
fi

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${REPO_NAME}"

echo "1. Create a new Space at: https://huggingface.co/new-space"
echo "   - Name: ${REPO_NAME}"
echo "   - SDK: Docker"
echo "   - Hardware: ZeroGPU (free A100)"
echo ""

echo "2. Clone the Space:"
echo "   git clone https://huggingface.co/spaces/${HF_USERNAME}/${REPO_NAME}"
echo "   cd ${REPO_NAME}"
echo ""

echo "3. Copy files from hf-space/ to your Space repo:"
echo "   cp ../hf-space/* ."
echo ""

echo "4. Push to HuggingFace:"
echo "   git add ."
echo "   git commit -m 'Deploy LLAVA-LENS'"
echo "   git push"
echo ""

echo "5. Your Space will be available at:"
echo "   ${SPACE_URL}"
echo ""

echo "6. To connect your custom domain:"
echo "   a. Go to Space Settings > Custom domain"
echo "   b. Add your domain (e.g., lens.yourdomain.com)"
echo "   c. Add CNAME record in your DNS:"
echo "      lens.yourdomain.com -> ${HF_USERNAME}-${REPO_NAME}.hf.space"
echo "   d. Enable HTTPS in Space Settings"
echo ""

echo "=== Done! ==="

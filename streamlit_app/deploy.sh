#!/bin/bash
# Deployment script for Twi-Chinese Translator

set -e

echo "🚀 Deploying Twi-Chinese Translator..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if model exists
if [ ! -f "../Attention_is_All_You_Need/results/model_best.ckpt" ]; then
    echo "⚠️  Warning: Model file not found at expected location"
    echo "   Expected: ../Attention_is_All_You_Need/results/model_best.ckpt"
    echo "   Please ensure model is trained and saved."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=twi-translator)" ]; then
    echo "🛑 Stopping existing container..."
    docker stop twi-translator 2>/dev/null || true
    docker rm twi-translator 2>/dev/null || true
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t twi-chinese-translator:latest .

# Run container with volume mounts
echo "🎯 Starting container..."
docker run -d \
    --name twi-translator \
    -p 8501:8501 \
    -v "$(pwd)/../Attention_is_All_You_Need/results:/app/models:ro" \
    -v "$(pwd)/../Attention_is_All_You_Need/data:/app/data:ro" \
    --restart unless-stopped \
    twi-chinese-translator:latest

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 5

# Check if container is running
if [ "$(docker ps -q -f name=twi-translator)" ]; then
    echo "✅ Container is running!"
    
    # Get container IP
    CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' twi-translator)
    HOST_IP=$(hostname -I | awk '{print $1}')
    
    echo ""
    echo "🌐 Access the app at:"
    echo "   Local:     http://localhost:8501"
    echo "   Container: http://${CONTAINER_IP}:8501"
    echo "   Network:   http://${HOST_IP}:8501"
    echo ""
    echo "📊 View logs:"
    echo "   docker logs -f twi-translator"
    echo ""
    
    # Show recent logs
    echo "🔍 Recent logs:"
    docker logs --tail 20 twi-translator 2>&1 || true
else
    echo "❌ Container failed to start!"
    echo "🔍 Checking logs..."
    docker logs twi-translator 2>&1 || true
    exit 1
fi

#!/bin/bash
# Docker Startup Script
# =====================
# Starts the Honey Bee Toxicity Prediction System using Docker Compose
#
# Usage: ./docker-start.sh

echo "🐝 Starting Honey Bee Toxicity Prediction System"
echo "=================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose"
    exit 1
fi

# Check if required files exist
if [ ! -f "outputs/models/best_model_xgboost.pkl" ]; then
    echo "❌ Error: Model file not found"
    echo "Please train the model first by running: python train_models_fast.py"
    exit 1
fi

if [ ! -f "outputs/preprocessors/preprocessor.pkl" ]; then
    echo "❌ Error: Preprocessor file not found"
    echo "Please run preprocessing first"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo "✓ Model and preprocessor files found"
echo ""

# Build and start containers
echo "📦 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ System started successfully!"
echo ""
echo "📍 Services available at:"
echo "   - API Backend:  http://localhost:8000"
echo "   - API Docs:     http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/health"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
echo ""


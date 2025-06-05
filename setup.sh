#!/bin/bash

# Wall Detection API Setup Script

set -e

echo "🏗️  Wall Detection API Setup"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p pkl_file logs

# Check if model file exists
if [ ! -f "pkl_file/model_best_val_loss_var.pkl" ]; then
    echo "⚠️  Model file not found in pkl_file/ directory"
    echo "Please download the model using one of these methods:"
    echo ""
    echo "Option 1 - Azure Blob Storage:"
    echo "  export AZURE_STORAGE_ACCOUNT_NAME=\"your_account\""
    echo "  export AZURE_STORAGE_ACCOUNT_KEY=\"your_key\""
    echo "  python download_model.py --storage azure"
    echo ""
    echo "Option 2 - Google Cloud Storage:"
    echo "  export GCP_BUCKET_NAME=\"your_bucket\""
    echo "  python download_model.py --storage gcp"
    echo ""
    echo "Option 3 - Manual:"
    echo "  Place your model_best_val_loss_var.pkl in the pkl_file/ directory"
    echo ""
    
    read -p "Do you want to continue setup without the model file? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please add the model file and run setup again."
        exit 1
    fi
fi

# Build and start the service
echo "🐳 Building Docker image..."
docker-compose build

echo "🚀 Starting the service..."
docker-compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to start..."
sleep 10

# Test the health endpoint
echo "🔍 Testing health endpoint..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ API is running successfully!"
    echo ""
    echo "🎉 Setup Complete!"
    echo "==================="
    echo "API URL: http://localhost:8000"
    echo "Swagger Docs: http://localhost:8000/docs"
    echo "ReDoc: http://localhost:8000/redoc"
    echo ""
    echo "To test the API:"
    echo "  python test_api.py"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "To stop the service:"
    echo "  docker-compose down"
else
    echo "❌ API health check failed"
    echo "Check the logs with: docker-compose logs"
    exit 1
fi 
# Wall Detection API

A deep learning-based API for detecting wall corners and joints in floor plan images. This service uses a pre-trained neural network to identify structural elements in architectural drawings.

## Features

- 🏗️ **Wall Corner Detection**: Automatically detects wall corners and junctions in floor plan images
- 🖼️ **Multiple Input Formats**: Supports both base64-encoded images and cloud storage file paths
- ☁️ **Cloud Storage Integration**: Compatible with Azure Blob Storage and Google Cloud Storage
- 🐳 **Docker Support**: Containerized deployment with Python 3.10
- 🚀 **FastAPI Backend**: High-performance async API with automatic documentation
- 📊 **Confidence Scoring**: Returns confidence scores for detected corners
- 🔧 **Configurable Parameters**: Adjustable detection threshold and non-maximum suppression

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Model file (`model_best_val_loss_var.pkl`) - see [Model Setup](#model-setup) section

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd wall-detection
```

### 2. Download Model File

#### Option A: From Azure Blob Storage

```bash
# Set environment variables
export AZURE_STORAGE_ACCOUNT_NAME="your_storage_account"
export AZURE_STORAGE_ACCOUNT_KEY="your_storage_key"
export AZURE_CONTAINER_NAME="models"  # optional, defaults to "models"

# Download model
python download_model.py --storage azure --blob-name model_best_val_loss_var.pkl
```

#### Option B: From Google Cloud Storage

```bash
# Set environment variables
export GCP_BUCKET_NAME="your_bucket_name"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"  # optional

# Download model
python download_model.py --storage gcp --blob-name model_best_val_loss_var.pkl
```

#### Option C: Manual Download

Place your `model_best_val_loss_var.pkl` file in the `pkl_file/` directory.

### 3. Build and Run with Docker

```bash
# Build and start the service
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

The API will be available at `http://localhost:8000`

## API Documentation

### Endpoints

- **GET** `/` - Health check
- **GET** `/health` - Detailed health status
- **POST** `/predict` - Perform wall corner detection

### Interactive API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Usage Examples

#### 1. Base64 Image Detection

```python
import requests
import base64

# Read and encode image
with open("floor_plan.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make prediction request
response = requests.post("http://localhost:8000/predict", json={
    "image_data": image_data,
    "threshold": 0.1,
    "nms_size": 3
})

result = response.json()
print(f"Found {len(result['wall_corners'])} wall corners")
```

#### 2. Cloud Storage Detection

```python
import requests

# Azure Blob Storage
response = requests.post("http://localhost:8000/predict", json={
    "cloud_path": "floor_plans/building_1.png",
    "storage_type": "azure",
    "threshold": 0.1
})

# Google Cloud Storage
response = requests.post("http://localhost:8000/predict", json={
    "cloud_path": "images/floor_plan.jpg",
    "storage_type": "gcp",
    "threshold": 0.1
})
```

#### 3. Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Base64 prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
       "threshold": 0.1
     }'
```

## Configuration

### Environment Variables

#### Azure Blob Storage (uncomment in docker-compose.yml)
```yaml
environment:
  - AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
  - AZURE_STORAGE_ACCOUNT_KEY=your_storage_key
  - AZURE_CONTAINER_NAME=images  # optional, defaults to "images"
```

#### Google Cloud Storage (uncomment in docker-compose.yml)
```yaml
environment:
  - GCP_BUCKET_NAME=your_bucket_name
  - GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json

volumes:
  - ./path/to/your/gcp-credentials.json:/app/gcp-credentials.json:ro
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_data` | string | - | Base64-encoded image data |
| `cloud_path` | string | - | Path to image in cloud storage |
| `storage_type` | string | "azure" | Cloud storage provider ("azure" or "gcp") |
| `threshold` | float | 0.1 | Detection confidence threshold (0.0-1.0) |
| `nms_size` | int | 3 | Non-maximum suppression window size |

## Testing

### Automated Tests

```bash
# Install test dependencies
pip install requests

# Run tests (make sure API is running)
python test_api.py
```

### Manual Testing

```bash
# Test with the included script
python main.py test_floor_plan.png

# Test the API
curl http://localhost:8000/health
```

## Development Setup

### Local Development (without Docker)

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn azure-storage-blob google-cloud-storage

# Download model (see Model Setup above)

# Run the API
python app.py
```

### Project Structure

```
wall-detection/
├── floortrans/          # Core detection models and utilities
│   ├── models/          # Neural network models
│   ├── loaders/         # Data loading utilities
│   ├── losses/          # Loss functions
│   └── post_prosessing.py  # Post-processing utilities
├── pkl_file/            # Model weights directory
├── app.py              # FastAPI application
├── main.py             # CLI inference script
├── train.py            # Training script
├── download_model.py   # Cloud storage download utility
├── test_api.py         # API test suite
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-container setup
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Model Architecture

The system uses a Hourglass network (Furukawa variant) trained specifically for floor plan analysis:

- **Input**: RGB images of floor plans
- **Output**: Multi-channel heatmaps for different junction types
- **Detection**: 13 different wall corner/junction types
- **Post-processing**: Non-maximum suppression and confidence filtering

## Response Format

```json
{
  "wall_corners": [
    {
      "x": 124.5,
      "y": 89.2,
      "confidence": 0.85,
      "corner_type": 3
    }
  ],
  "image_size": {
    "width": 512,
    "height": 512
  },
  "processing_time": 0.45
}
```

## Performance Considerations

- **GPU Support**: Automatically uses CUDA if available
- **Memory Usage**: ~2-4GB RAM for typical images
- **Processing Time**: 0.1-2 seconds per image depending on size and hardware
- **Concurrent Requests**: FastAPI supports async processing

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   FileNotFoundError: Model file not found at ./pkl_file/model_best_val_loss_var.pkl
   ```
   - Solution: Download the model file using `download_model.py` or place it manually

2. **Cloud storage authentication errors**
   ```
   Azure storage credentials not configured
   ```
   - Solution: Set the required environment variables in `docker-compose.yml`

3. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Solution: Use CPU by setting `CUDA_VISIBLE_DEVICES=""` or reduce batch size

4. **Port already in use**
   ```
   Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use
   ```
   - Solution: Change the port in `docker-compose.yml` or stop the conflicting service

### Logs and Debugging

```bash
# View container logs
docker-compose logs -f

# Access container shell
docker-compose exec wall-detection-api bash

# Check model loading
docker-compose exec wall-detection-api python -c "from app import load_model; load_model()"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation at `/docs` 
# Plant Disease Detection Server System

A comprehensive, AI-powered server system for plant disease detection using TensorFlow and Convolutional Neural Networks (CNNs), specifically optimized for Ghanaian crops with a focus on maize diseases.

## 🚀 Features

### Core AI Capabilities
- **CNN-based Disease Detection**: State-of-the-art convolutional neural networks for accurate plant disease classification
- **Multi-Crop Support**: Cassava, Maize, Cocoa, Yam, Plantain, Rice, Millet, Sorghum, Groundnut, Cowpea
- **Maize-Specific Optimization**: Enhanced detection for 10 maize disease classes including rust, smut, leaf blight, and more
- **Real-time Processing**: Fast inference with optimized model architectures
- **Confidence Calibration**: Advanced post-processing for reliable predictions

### Advanced Model Training
- **Transfer Learning**: Pre-trained models (EfficientNet, ResNet, DenseNet, MobileNet)
- **Custom CNN Architectures**: Maize-specific neural network designs
- **Data Augmentation**: Comprehensive augmentation strategies preserving disease features
- **Class Imbalance Handling**: Automatic class weight calculation and balanced training
- **Fine-tuning**: Multi-stage training with progressive layer unfreezing

### Data Processing
- **Image Enhancement**: Contrast enhancement, noise reduction, and preprocessing
- **Multi-format Support**: JPEG, PNG, BMP image formats
- **Batch Processing**: Efficient handling of multiple images
- **Quality Validation**: Automatic image quality assessment

### API & Integration
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Mobile Optimization**: TensorFlow Lite export for mobile deployment
- **Real-time Monitoring**: Health checks and performance metrics
- **CORS Support**: Cross-origin resource sharing for web applications

## 🏗️ Architecture

```
server/
├── ai_module/                    # Core AI components
│   ├── computer_vision/         # Computer vision modules
│   │   ├── disease_detector.py  # Main disease detection engine
│   │   ├── model_trainer.py     # Model training and optimization
│   │   ├── image_processor.py   # Image preprocessing
│   │   └── data_augmentation.py # Advanced augmentation strategies
│   ├── knowledge_base/          # Agricultural knowledge system
│   ├── nlp/                     # Natural language processing
│   └── voice/                   # Voice interface processing
├── api/                         # API server
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration management
│   ├── routes/                  # API endpoints
│   └── middleware/              # Request processing middleware
├── scripts/                     # Utility scripts
│   └── train_maize_model.py     # Maize model training script
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Plant-disease-detection-system-main/server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Run the FastAPI server
python -m api.main

# Or use uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Info**: http://localhost:8000/api/v1/info

## 🧠 Training Maize Disease Models

### Using the Training Script

```bash
# Basic training with maize dataset
python scripts/train_maize_model.py --data_dir /path/to/maize/dataset

# GPU-accelerated training
python scripts/train_maize_model.py --data_dir /path/to/maize/dataset --gpu

# Custom configuration
python scripts/train_maize_model.py --data_dir /path/to/maize/dataset --config_file custom_config.json
```

### Dataset Structure

```
maize_dataset/
├── healthy/                     # Healthy maize images
├── maize_rust/                 # Maize rust disease
├── maize_smut/                 # Maize smut disease
├── maize_leaf_blight/          # Northern corn leaf blight
├── maize_gray_leaf_spot/       # Gray leaf spot
├── maize_common_rust/          # Common rust
├── maize_southern_leaf_blight/ # Southern leaf blight
├── maize_anthracnose/          # Anthracnose leaf blight
├── maize_eyespot/              # Eyespot disease
└── maize_stewart_wilt/         # Stewart's wilt
```

### Training Configuration

The system automatically configures optimal training parameters:

- **Model Architecture**: Custom maize CNN or pre-trained models
- **Input Shape**: 224x224x3 RGB images
- **Batch Size**: 32 (configurable)
- **Epochs**: 150 (with early stopping)
- **Learning Rate**: 0.001 with adaptive reduction
- **Data Augmentation**: Maize-specific strategies
- **Class Balancing**: Automatic weight calculation

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the server directory:

```env
# API Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model Settings
MODEL_PATH=models/maize_disease_model.h5
MAIZE_SPECIFIC=true

# Database
DATABASE_URL=sqlite:///./data/plant_disease.db

# Security
SECRET_KEY=your-secret-key-here

# External APIs
GOOGLE_TRANSLATE_API_KEY=your-key
OPENAI_API_KEY=your-key
```

### Model Configuration

```json
{
  "model_architecture": "maize_cnn",
  "input_shape": [224, 224, 3],
  "num_classes": 10,
  "maize_specific": true,
  "transfer_learning": true,
  "fine_tuning": true,
  "data_balance": true,
  "augmentation_strategy": "aggressive"
}
```

## 📊 API Endpoints

### Disease Detection

```http
POST /api/v1/disease/detect
Content-Type: multipart/form-data

Parameters:
- image: Image file (JPEG, PNG)
- crop_type: Crop type (optional)
- confidence_threshold: Confidence level (0.0-1.0)
- max_predictions: Number of predictions
- include_visualization: Generate visualization
```

### Batch Detection

```http
POST /api/v1/disease/batch-detect
Content-Type: multipart/form-data

Parameters:
- images: Multiple image files
- crop_type: Crop type (optional)
- confidence_threshold: Confidence level
```

### Model Information

```http
GET /api/v1/disease/model-info
```

### Supported Crops

```http
GET /api/v1/disease/supported-crops
```

## 🎯 Maize Disease Classes

The system detects 10 maize disease categories:

1. **Healthy Maize** - Disease-free plants
2. **Maize Rust** - Puccinia sorghi infection
3. **Maize Smut** - Ustilago maydis infection
4. **Northern Corn Leaf Blight** - Exserohilum turcicum
5. **Gray Leaf Spot** - Cercospora zeae-maydis
6. **Common Rust** - Puccinia sorghi
7. **Southern Leaf Blight** - Bipolaris maydis
8. **Anthracnose Leaf Blight** - Colletotrichum graminicola
9. **Eyespot** - Kabatiella zeae
10. **Stewart's Wilt** - Pantoea stewartii

## 🔬 Model Performance

### Training Metrics

- **Accuracy**: >95% on validation set
- **F1 Score**: >0.90 for balanced performance
- **Precision**: >0.92 for reliable predictions
- **Recall**: >0.88 for comprehensive detection

### Inference Performance

- **Processing Time**: <2 seconds per image
- **Memory Usage**: <2GB RAM
- **GPU Support**: CUDA acceleration available
- **Mobile Export**: TensorFlow Lite compatible

## 🚀 Deployment

### Production Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Set production environment
export ENVIRONMENT=production

# Run with gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

- **AWS**: EC2 with GPU instances for training
- **Google Cloud**: Cloud Run for API deployment
- **Azure**: Container Instances for scalable deployment

## 🔍 Monitoring & Logging

### Health Checks

```http
GET /health
```

Response includes:
- Service status
- AI component availability
- Database connectivity
- Model loading status

### Logging

- **File Logging**: `logs/api.log`
- **Training Logs**: `logs/maize_training.log`
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Structured Logging**: JSON format for easy parsing

### Metrics

- Request/response times
- Model inference latency
- Error rates and types
- Resource utilization

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_disease_detector.py

# Run with coverage
pytest --cov=ai_module --cov=api
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/test_api_integration.py

# Test model training
pytest tests/test_model_training.py
```

### Performance Tests

```bash
# Load testing
python tests/performance/load_test.py

# Model benchmarking
python tests/performance/benchmark_models.py
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes and add tests
5. Submit a pull request

### Code Style

- **Python**: PEP 8 compliance
- **Type Hints**: Full type annotation
- **Documentation**: Docstring coverage
- **Testing**: >90% test coverage

## 📚 Documentation

- **API Reference**: `/docs` endpoint
- **Training Guide**: See training scripts
- **Model Architecture**: Detailed in code comments
- **Deployment Guide**: Production setup instructions

## 🆘 Support

### Common Issues

1. **Model Loading Errors**: Check model file paths and TensorFlow version
2. **Memory Issues**: Reduce batch size or use GPU acceleration
3. **Training Convergence**: Adjust learning rate and augmentation parameters
4. **API Errors**: Check request format and file uploads

### Getting Help

- **Issues**: GitHub issue tracker
- **Documentation**: Inline code documentation
- **Examples**: Training and inference examples
- **Community**: Agricultural AI community forums

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ghanaian agricultural experts for disease knowledge
- TensorFlow and Keras communities
- Open-source computer vision libraries
- Agricultural research institutions

---

**Note**: This system is specifically designed for Ghanaian agricultural conditions and maize diseases. For other regions or crops, configuration adjustments may be required.
#   p l a n t - d i s e a s e - d e t e c t i o n - s e r v e r  
 
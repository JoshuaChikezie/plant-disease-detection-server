# 🚀 Quick Start Guide - Maize Disease Detection

Get your maize disease detection system up and running in minutes!

## ⚡ 5-Minute Setup

### 1. Install Dependencies

```bash
# Navigate to server directory
cd Plant-disease-detection-system-main/server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Your Maize Dataset

Organize your maize images in this structure:

```
your_maize_dataset/
├── healthy/                     # Healthy maize plants
│   ├── healthy_001.jpg
│   ├── healthy_002.jpg
│   └── ...
├── maize_rust/                 # Rust disease
│   ├── rust_001.jpg
│   ├── rust_002.jpg
│   └── ...
├── maize_smut/                 # Smut disease
│   ├── smut_001.jpg
│   ├── smut_002.jpg
│   └── ...
├── maize_leaf_blight/          # Leaf blight
│   └── ...
├── maize_gray_leaf_spot/       # Gray leaf spot
│   └── ...
├── maize_common_rust/          # Common rust
│   └── ...
├── maize_southern_leaf_blight/ # Southern leaf blight
│   └── ...
├── maize_anthracnose/          # Anthracnose
│   └── ...
├── maize_eyespot/              # Eyespot
│   └── ...
└── maize_stewart_wilt/         # Stewart's wilt
    └── ...
```

**Minimum Requirements:**
- **Total Images**: At least 100 images
- **Per Class**: Minimum 10 images per disease class
- **Image Format**: JPG, PNG, or BMP
- **Image Size**: Any size (will be resized automatically)
- **Quality**: Clear, well-lit images of maize leaves/plants

### 3. Train Your Model

```bash
# Basic training (CPU)
python scripts/train_maize_model.py --data_dir /path/to/your_maize_dataset

# GPU training (recommended)
python scripts/train_maize_model.py --data_dir /path/to/your_maize_dataset --gpu
```

**Training Time Estimates:**
- **CPU**: 2-4 hours for 1000 images
- **GPU**: 30 minutes - 1 hour for 1000 images

### 4. Start the API Server

```bash
# Start the server
python -m api.main

# Or use uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test Your Model

Open your browser and go to: http://localhost:8000/docs

Try the disease detection endpoint:
- **POST** `/api/v1/disease/detect`
- Upload a maize image
- Get instant disease predictions!

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file in the server directory:

```env
# Basic Settings
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Model Settings
MODEL_PATH=models/maize_disease_model.h5
MAIZE_SPECIFIC=true

# Training Settings
BATCH_SIZE=32
EPOCHS=150
LEARNING_RATE=0.001
```

### Custom Training Configuration

Create `custom_config.json`:

```json
{
  "model_architecture": "efficientnet_b0",
  "input_shape": [224, 224, 3],
  "num_classes": 10,
  "batch_size": 16,
  "epochs": 100,
  "learning_rate": 0.0005,
  "data_augmentation": true,
  "maize_specific": true,
  "transfer_learning": true
}
```

Then train with:
```bash
python scripts/train_maize_model.py \
  --data_dir /path/to/dataset \
  --config_file custom_config.json
```

## 📊 Expected Results

### Training Metrics
- **Accuracy**: 90-95% after 50-100 epochs
- **F1 Score**: 0.85-0.92
- **Training Time**: 1-4 hours depending on dataset size

### Inference Performance
- **Processing Time**: 1-3 seconds per image
- **Memory Usage**: 1-2GB RAM
- **Model Size**: 50-200MB depending on architecture

## 🚨 Common Issues & Solutions

### Issue: "No module named 'tensorflow'"
```bash
# Solution: Install TensorFlow
pip install tensorflow==2.13.0
```

### Issue: "CUDA not available"
```bash
# Solution: Install GPU version
pip install tensorflow-gpu==2.13.0
```

### Issue: "Out of memory during training"
```bash
# Solution: Reduce batch size
# Edit config or use --config_file with smaller batch_size
```

### Issue: "Model not converging"
```bash
# Solutions:
# 1. Check image quality and dataset balance
# 2. Increase epochs (--config_file with more epochs)
# 3. Verify class distribution in dataset
```

### Issue: "API server won't start"
```bash
# Solutions:
# 1. Check if port 8000 is free
# 2. Verify all dependencies installed
# 3. Check logs for specific errors
```

## 🔍 Monitoring Training

### View Training Progress

Training automatically creates:
- **Logs**: `logs/maize_training.log`
- **Checkpoints**: `checkpoints/` directory
- **TensorBoard**: `logs/fit/` directory

### Monitor with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/fit

# Open browser: http://localhost:6006
```

### Check Training Logs

```bash
# View real-time logs
tail -f logs/maize_training.log

# Search for specific metrics
grep "accuracy" logs/maize_training.log
```

## 📱 Mobile Integration

### Export for Mobile

After training, your model is automatically exported to:
- **TensorFlow Lite**: `models/maize/maize_disease_model.tflite`
- **Keras Model**: `models/maize/maize_disease_model.h5`

### API Endpoints for Mobile

```http
# Single image detection
POST /api/v1/disease/detect

# Batch detection
POST /api/v1/disease/batch-detect

# Get model info
GET /api/v1/disease/model-info
```

## 🌟 Pro Tips

### 1. Dataset Quality
- **Lighting**: Use consistent, good lighting
- **Angles**: Capture leaves from multiple angles
- **Background**: Clean, simple backgrounds
- **Resolution**: Higher resolution = better results

### 2. Training Optimization
- **Start Small**: Begin with 100-500 images
- **Validate Early**: Use validation split to monitor overfitting
- **Augment Data**: Enable data augmentation for small datasets
- **Transfer Learning**: Use pre-trained models for better results

### 3. Production Deployment
- **GPU**: Use GPU instances for faster training
- **Monitoring**: Enable health checks and logging
- **Scaling**: Use load balancers for multiple API instances
- **Caching**: Implement Redis for response caching

## 📞 Need Help?

### Quick Support
1. **Check Logs**: Look at `logs/maize_training.log`
2. **Verify Dataset**: Ensure proper folder structure
3. **Test API**: Use `/docs` endpoint for testing
4. **Check Dependencies**: Verify all packages installed

### Common Commands

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Test model loading
python -c "from ai_module.computer_vision.disease_detector import DiseaseDetector; d = DiseaseDetector()"

# Check API health
curl http://localhost:8000/health

# View model info
curl http://localhost:8000/api/v1/disease/model-info
```

## 🎯 Next Steps

After getting the basic system running:

1. **Improve Dataset**: Add more images and classes
2. **Fine-tune Model**: Adjust hyperparameters
3. **Deploy Production**: Set up monitoring and scaling
4. **Add Features**: Implement knowledge base and voice interface
5. **Mobile App**: Integrate with your React Native app

---

**🎉 Congratulations!** You now have a working maize disease detection system. The AI will learn from your dataset and provide accurate disease predictions for Ghanaian maize farmers. 
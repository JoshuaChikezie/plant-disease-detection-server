#!/usr/bin/env python3
"""
Test Script for Maize Disease Detection System

This script tests the core functionality of the system to ensure
everything is working correctly before training or deployment.
"""

import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        # Test core imports
        import tensorflow as tf
        logger.info(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        import cv2
        logger.info(f"✅ OpenCV {cv2.__version__} imported successfully")
        
        import numpy as np
        logger.info(f"✅ NumPy {np.__version__} imported successfully")
        
        # Test custom modules
        from ai_module.computer_vision.disease_detector import DiseaseDetector
        logger.info("✅ DiseaseDetector imported successfully")
        
        from ai_module.computer_vision.model_trainer import ModelTrainer
        logger.info("✅ ModelTrainer imported successfully")
        
        from ai_module.computer_vision.data_augmentation import DataAugmentation
        logger.info("✅ DataAugmentation imported successfully")
        
        from ai_module.computer_vision.image_processor import ImageProcessor
        logger.info("✅ ImageProcessor imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_gpu():
    """Test GPU availability."""
    logger.info("Testing GPU availability...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✅ GPU found: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"   - {gpu.name}")
            return True
        else:
            logger.info("⚠️  No GPU found, will use CPU")
            return True
            
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False

def test_disease_detector():
    """Test disease detector initialization."""
    logger.info("Testing disease detector...")
    
    try:
        from ai_module.computer_vision.disease_detector import DiseaseDetector
        
        # Initialize detector
        detector = DiseaseDetector()
        logger.info("✅ DiseaseDetector initialized successfully")
        
        # Check configuration
        config = detector.config
        logger.info(f"✅ Configuration loaded: {config['model_architecture']}")
        
        # Check disease classes
        maize_diseases = detector.disease_classes.get('maize', {})
        logger.info(f"✅ Maize diseases configured: {len(maize_diseases)} classes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Disease detector test failed: {e}")
        return False

def test_model_trainer():
    """Test model trainer initialization."""
    logger.info("Testing model trainer...")
    
    try:
        from ai_module.computer_vision.model_trainer import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer()
        logger.info("✅ ModelTrainer initialized successfully")
        
        # Check configuration
        config = trainer.config
        logger.info(f"✅ Training config: {config['model_architecture']}")
        logger.info(f"✅ Batch size: {config['batch_size']}")
        logger.info(f"✅ Epochs: {config['epochs']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model trainer test failed: {e}")
        return False

def test_data_augmentation():
    """Test data augmentation functionality."""
    logger.info("Testing data augmentation...")
    
    try:
        from ai_module.computer_vision.data_augmentation import DataAugmentation
        
        # Initialize augmenter
        augmenter = DataAugmentation()
        logger.info("✅ DataAugmentation initialized successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        logger.info(f"✅ Test image created: {test_image.shape}")
        
        # Test basic augmentation
        augmented = augmenter.augment_image(test_image, 'basic')
        logger.info(f"✅ Basic augmentation applied: {augmented.shape}")
        
        # Test maize-specific augmentation
        maize_augmented = augmenter.augment_image(test_image, 'maize')
        logger.info(f"✅ Maize augmentation applied: {maize_augmented.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data augmentation test failed: {e}")
        return False

def test_image_processor():
    """Test image processor functionality."""
    logger.info("Testing image processor...")
    
    try:
        from ai_module.computer_vision.image_processor import ImageProcessor
        
        # Initialize processor
        processor = ImageProcessor()
        logger.info("✅ ImageProcessor initialized successfully")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test resize
        resized = processor.resize_image(test_image, (224, 224))
        logger.info(f"✅ Image resize: {test_image.shape} -> {resized.shape}")
        
        # Test normalization
        normalized = processor.normalize_image(resized)
        logger.info(f"✅ Image normalization: range [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Image processor test failed: {e}")
        return False

def test_directory_structure():
    """Test if required directories exist."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        'models',
        'logs',
        'data',
        'uploads',
        'static'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✅ Directory exists: {dir_name}")
        else:
            logger.info(f"⚠️  Directory missing: {dir_name}")
            missing_dirs.append(dir_name)
    
    # Create missing directories
    if missing_dirs:
        logger.info("Creating missing directories...")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_name}")
    
    return True

def test_api_imports():
    """Test API module imports."""
    logger.info("Testing API imports...")
    
    try:
        from api.config import settings
        logger.info("✅ API config imported successfully")
        
        from api.main import app
        logger.info("✅ FastAPI app imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API import test failed: {e}")
        return False

def create_sample_image():
    """Create a sample maize image for testing."""
    logger.info("Creating sample maize image...")
    
    try:
        # Create a simple green image (simulating maize leaf)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add green color (maize leaf)
        image[:, :, 1] = 150  # Green channel
        
        # Add some texture (simulating leaf veins)
        for i in range(0, 224, 20):
            image[:, i:i+2, 1] = 200  # Darker green lines
        
        # Save the image
        sample_path = "test_maize_sample.jpg"
        cv2.imwrite(sample_path, image)
        logger.info(f"✅ Sample image created: {sample_path}")
        
        return sample_path
        
    except Exception as e:
        logger.error(f"❌ Sample image creation failed: {e}")
        return None

def run_all_tests():
    """Run all system tests."""
    logger.info("🚀 Starting system tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("Directory Structure", test_directory_structure),
        ("Disease Detector", test_disease_detector),
        ("Model Trainer", test_model_trainer),
        ("Data Augmentation", test_data_augmentation),
        ("Image Processor", test_image_processor),
        ("API Imports", test_api_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False

def main():
    """Main test function."""
    logger.info("🌱 Maize Disease Detection System - System Test")
    logger.info("=" * 60)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        # Create sample image
        sample_path = create_sample_image()
        if sample_path:
            logger.info(f"\n📸 Sample image created for testing: {sample_path}")
            logger.info("You can use this image to test the disease detection API")
        
        logger.info("\n🚀 Next steps:")
        logger.info("1. Prepare your maize dataset")
        logger.info("2. Run: python scripts/train_maize_model.py --data_dir /path/to/dataset")
        logger.info("3. Start the API: python -m api.main")
        logger.info("4. Test at: http://localhost:8000/docs")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

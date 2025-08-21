#!/usr/bin/env python3
"""
Maize Disease Detection Model Training Script

This script trains CNN models specifically for maize disease detection
using the enhanced ModelTrainer class with maize-specific optimizations.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_module.computer_vision.model_trainer import ModelTrainer
from ai_module.computer_vision.data_augmentation import DataAugmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/maize_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_maize_training_config():
    """Create maize-specific training configuration."""
    return {
        'model_architecture': 'maize_cnn',  # Use custom maize CNN
        'input_shape': (224, 224, 3),
        'num_classes': 10,  # Maize diseases + healthy
        'batch_size': 32,
        'epochs': 150,  # More epochs for maize diseases
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'data_augmentation': True,
        'early_stopping': True,
        'model_checkpoint': True,
        'reduce_lr_on_plateau': True,
        'class_weights': None,  # Will be calculated automatically
        'optimizer': 'adam',
        'loss_function': 'categorical_crossentropy',
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
        'callbacks': ['early_stopping', 'model_checkpoint', 'reduce_lr'],
        'save_best_only': True,
        'monitor': 'val_f1_score',  # Monitor F1 score for better balance
        'patience': 20,  # Increased patience for maize diseases
        'min_delta': 0.001,
        'maize_specific': True,  # Enable maize-specific optimizations
        'transfer_learning': True,  # Use pre-trained models
        'fine_tuning': True,  # Fine-tune pre-trained models
        'data_balance': True,  # Handle class imbalance
        'cross_validation': False,  # Enable for small datasets
        'ensemble_training': False,  # Train multiple models
        'augmentation_strategy': 'aggressive'  # More aggressive augmentation
    }


def setup_directories():
    """Create necessary directories for training."""
    dirs = [
        'models',
        'logs',
        'data/maize',
        'results',
        'checkpoints'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def validate_dataset(data_dir: str) -> bool:
    """Validate the maize dataset structure."""
    if not os.path.exists(data_dir):
        logger.error(f"Dataset directory does not exist: {data_dir}")
        return False
    
    # Check for expected class directories
    expected_classes = [
        'healthy',
        'maize_rust',
        'maize_smut', 
        'maize_leaf_blight',
        'maize_gray_leaf_spot',
        'maize_common_rust',
        'maize_southern_leaf_blight',
        'maize_anthracnose',
        'maize_eyespot',
        'maize_stewart_wilt'
    ]
    
    found_classes = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    
    missing_classes = set(expected_classes) - set(found_classes)
    if missing_classes:
        logger.warning(f"Missing classes: {missing_classes}")
    
    # Check if we have enough data
    total_images = 0
    for class_dir in found_classes:
        class_path = os.path.join(data_dir, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_images += len(images)
        logger.info(f"Class {class_dir}: {len(images)} images")
    
    logger.info(f"Total images found: {total_images}")
    
    if total_images < 100:
        logger.warning("Dataset seems small. Consider using data augmentation.")
    
    return True


def train_maize_model(data_dir: str, output_dir: str, config: dict):
    """Train the maize disease detection model."""
    logger.info("Starting maize disease detection model training...")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Load data
    logger.info("Loading training data...")
    try:
        images, labels, class_names = trainer.load_data(data_dir, crop_type='maize')
        logger.info(f"Data loaded successfully: {images.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Create and train model
    logger.info("Creating model architecture...")
    model = trainer.create_model()
    
    # Train model
    logger.info("Starting model training...")
    model_save_path = os.path.join(output_dir, 'maize_disease_model.h5')
    
    try:
        history = trainer.train(images, labels, model_save_path=model_save_path)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
            images, labels, test_size=0.2, stratify=np.argmax(labels, axis=1), random_state=42
        )
        
        evaluation_results = trainer.evaluate(test_imgs, test_lbls)
        
        # Save evaluation results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Print key metrics
        logger.info(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        if 'per_class_metrics' in evaluation_results:
            logger.info("Per-class metrics:")
            for class_name, metrics in evaluation_results['per_class_metrics'].items():
                logger.info(f"  {class_name}: F1={metrics['f1_score']:.4f}, "
                          f"Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'model_config.json')
    trainer.save_model(model_save_path, config_path)
    
    # Generate visualizations
    logger.info("Generating training visualizations...")
    try:
        # Training history
        history_path = os.path.join(output_dir, 'training_history.png')
        trainer.plot_training_history(history_path)
        
        # Confusion matrix
        if 'confusion_matrix' in evaluation_results:
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            trainer.plot_confusion_matrix(evaluation_results['confusion_matrix'], cm_path)
        
        logger.info("Visualizations generated successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
    
    # Export for mobile deployment
    logger.info("Exporting model for mobile deployment...")
    try:
        tflite_path = os.path.join(output_dir, 'maize_disease_model.tflite')
        if trainer.export_model_for_mobile(tflite_path):
            logger.info(f"Model exported to TensorFlow Lite: {tflite_path}")
        else:
            logger.warning("Failed to export TensorFlow Lite model")
    except Exception as e:
        logger.error(f"Mobile export failed: {e}")
    
    logger.info("Training pipeline completed successfully!")
    return True


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Maize Disease Detection Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to maize dataset directory')
    parser.add_argument('--output_dir', type=str, default='models/maize',
                       help='Output directory for trained model')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU training')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Validate dataset
    if not validate_dataset(args.data_dir):
        logger.error("Dataset validation failed. Exiting.")
        return 1
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded custom configuration from: {args.config_file}")
    else:
        config = create_maize_training_config()
        logger.info("Using default maize training configuration")
    
    # GPU setup
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU training enabled. Found {len(gpus)} GPU(s)")
        else:
            logger.warning("GPU requested but none found. Using CPU.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    try:
        success = train_maize_model(args.data_dir, args.output_dir, config)
        if success:
            logger.info("Training completed successfully!")
            return 0
        else:
            logger.error("Training failed!")
            return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

"""
Model Trainer - Training CNN models for plant disease detection

This module provides comprehensive training capabilities for plant disease
detection models, including data loading, model architecture, training loops,
and evaluation metrics with focus on maize diseases.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime
import cv2 # Added for maize-specific preprocessing

from .image_processor import ImageProcessor
from .data_augmentation import DataAugmentation

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model trainer for plant disease detection.
    
    Provides functionality for:
    - Data loading and preprocessing
    - Model architecture definition
    - Training and validation
    - Model evaluation and metrics
    - Model saving and loading
    - Maize-specific optimizations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.image_processor = ImageProcessor()
        self.data_augmentation = DataAugmentation()
        
        # Training state
        self.model = None
        self.history = None
        self.class_names = []
        
        # Set up GPU if available
        self._setup_gpu()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration optimized for maize disease detection."""
        return {
            'model_architecture': 'efficientnet_b0',
            'input_shape': (224, 224, 3),
            'num_classes': 10,  # Increased for maize diseases
            'batch_size': 32,
            'epochs': 100,  # Increased for better convergence
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'data_augmentation': True,
            'early_stopping': True,
            'model_checkpoint': True,
            'reduce_lr_on_plateau': True,
            'class_weights': None,
            'optimizer': 'adam',
            'loss_function': 'categorical_crossentropy',
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'callbacks': ['early_stopping', 'model_checkpoint', 'reduce_lr'],
            'save_best_only': True,
            'monitor': 'val_f1_score',  # Monitor F1 score for better balance
            'patience': 15,  # Increased patience
            'min_delta': 0.001,
            'maize_specific': True,  # Flag for maize-focused training
            'transfer_learning': True,  # Use pre-trained models
            'fine_tuning': True,  # Fine-tune pre-trained models
            'data_balance': True,  # Handle class imbalance
            'cross_validation': False,  # Enable for small datasets
            'ensemble_training': False,  # Train multiple models
            'augmentation_strategy': 'aggressive'  # More aggressive augmentation for small datasets
        }
    
    def _setup_gpu(self):
        """Configure GPU settings for training."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU setup complete. Found {len(gpus)} GPU(s)")
                
                # Set mixed precision for better performance
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled")
            else:
                logger.info("No GPU found. Using CPU for training.")
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
    
    def load_data(self, data_dir: str, crop_type: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Enhanced data loading with maize-specific optimizations.
        
        Args:
            data_dir: Directory containing training data
            crop_type: Type of crop for filtering
            
        Returns:
            Tuple of (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        # Expected directory structure:
        # data_dir/
        #   ├── healthy/
        #   ├── disease_1/
        #   ├── disease_2/
        #   └── ...
        
        for class_dir in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            
            if os.path.isdir(class_path):
                class_names.append(class_dir)
                class_label = len(class_names) - 1
                
                # Load images from this class
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                logger.info(f"Loading {len(image_files)} images from class: {class_dir}")
                
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        # Load and preprocess image
                        image = self.image_processor.load_image(image_path)
                        image = self.image_processor.resize_image(
                            image, self.config['input_shape'][:2]
                        )
                        
                        # Apply initial preprocessing
                        if self.config.get('maize_specific', False):
                            image = self._apply_maize_specific_preprocessing(image)
                        
                        images.append(image)
                        labels.append(class_label)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {image_path}: {e}")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # One-hot encode labels
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
        
        logger.info(f"Loaded {len(images)} images from {len(class_names)} classes")
        logger.info(f"Class names: {class_names}")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        self.class_names = class_names
        return images, labels, class_names
    
    def _apply_maize_specific_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply maize-specific preprocessing techniques."""
        # Enhance contrast for better disease visibility
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return image
    
    def create_model(self, architecture: str = None) -> keras.Model:
        """
        Create enhanced CNN model architecture optimized for maize disease detection.
        
        Args:
            architecture: Model architecture name
            
        Returns:
            Compiled Keras model
        """
        architecture = architecture or self.config['model_architecture']
        
        if architecture.startswith('efficientnet'):
            model = self._create_efficientnet_model(architecture)
        elif architecture.startswith('resnet'):
            model = self._create_resnet_model(architecture)
        elif architecture.startswith('mobilenet'):
            model = self._create_mobilenet_model(architecture)
        elif architecture.startswith('densenet'):
            model = self._create_densenet_model(architecture)
        elif architecture == 'maize_cnn':
            model = self._create_maize_specific_cnn()
        else:
            model = self._create_custom_model()
        
        # Compile model with enhanced metrics
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss_function(),
            metrics=self._get_metrics()
        )
        
        self.model = model
        return model
    
    def _create_maize_specific_cnn(self) -> keras.Model:
        """Create a custom CNN specifically designed for maize disease detection."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.config['input_shape']),
            
            # First convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _create_efficientnet_model(self, architecture: str) -> keras.Model:
        """Create EfficientNet-based model with maize-specific optimizations."""
        # Load pre-trained EfficientNet
        if architecture == 'efficientnet_b0':
            base_model = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        elif architecture == 'efficientnet_b1':
            base_model = keras.applications.EfficientNetB1(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        elif architecture == 'efficientnet_b2':
            base_model = keras.applications.EfficientNetB2(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        else:
            base_model = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create model with maize-specific head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _create_densenet_model(self, architecture: str) -> keras.Model:
        """Create DenseNet-based model for maize disease detection."""
        if architecture == 'densenet121':
            base_model = keras.applications.DenseNet121(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        else:
            base_model = keras.applications.DenseNet121(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self, architecture: str) -> keras.Model:
        """Create ResNet-based model with maize-specific optimizations."""
        if architecture == 'resnet50':
            base_model = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        elif architecture == 'resnet101':
            base_model = keras.applications.ResNet101(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        else:
            base_model = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=self.config['input_shape']
            )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _create_mobilenet_model(self, architecture: str) -> keras.Model:
        """Create MobileNet-based model optimized for mobile deployment."""
        base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.config['input_shape']
        )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _create_custom_model(self) -> keras.Model:
        """Create custom CNN architecture with modern best practices."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.config['input_shape']),
            
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = self.config['optimizer']
        learning_rate = self.config['learning_rate']
        
        if optimizer_name == 'adam':
            return optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        elif optimizer_name == 'sgd':
            return optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer_name == 'rmsprop':
            return optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
        elif optimizer_name == 'adamw':
            return optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
        else:
            return optimizers.Adam(learning_rate=learning_rate)
    
    def _get_loss_function(self) -> str:
        """Get loss function based on configuration."""
        return self.config['loss_function']
    
    def _get_metrics(self) -> List[str]:
        """Get enhanced metrics for maize disease detection."""
        metrics = []
        for metric in self.config['metrics']:
            if metric == 'precision':
                metrics.append(keras.metrics.Precision())
            elif metric == 'recall':
                metrics.append(keras.metrics.Recall())
            elif metric == 'f1_score':
                metrics.append(self._f1_score_metric())
            else:
                metrics.append(metric)
        return metrics
    
    def _f1_score_metric(self):
        """Custom F1 score metric."""
        def f1_score(y_true, y_pred):
            precision = keras.metrics.Precision()(y_true, y_pred)
            recall = keras.metrics.Recall()(y_true, y_pred)
            return 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
        return f1_score
    
    def _get_callbacks(self, model_save_path: str = None) -> List[keras.callbacks.Callback]:
        """Get enhanced training callbacks."""
        callbacks_list = []
        
        if 'early_stopping' in self.config['callbacks'] and self.config['early_stopping']:
            early_stopping = callbacks.EarlyStopping(
                monitor=self.config['monitor'],
                patience=self.config['patience'],
                min_delta=self.config['min_delta'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)
        
        if 'model_checkpoint' in self.config['callbacks'] and self.config['model_checkpoint']:
            if model_save_path:
                checkpoint = callbacks.ModelCheckpoint(
                    filepath=model_save_path,
                    monitor=self.config['monitor'],
                    save_best_only=self.config['save_best_only'],
                    save_weights_only=False,
                    verbose=1
                )
                callbacks_list.append(checkpoint)
        
        if 'reduce_lr' in self.config['callbacks'] and self.config['reduce_lr_on_plateau']:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor=self.config['monitor'],
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
            callbacks_list.append(reduce_lr)
        
        # Add TensorBoard callback
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks_list.append(tensorboard_callback)
        
        # Add custom callback for maize-specific monitoring
        if self.config.get('maize_specific', False):
            maize_callback = self._MaizeTrainingCallback()
            callbacks_list.append(maize_callback)
        
        return callbacks_list
    
    class _MaizeTrainingCallback(keras.callbacks.Callback):
        """Custom callback for monitoring maize-specific training metrics."""
        
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                # Log maize-specific metrics
                if 'val_f1_score' in logs:
                    logger.info(f"Epoch {epoch + 1}: F1 Score = {logs['val_f1_score']:.4f}")
                if 'val_precision' in logs and 'val_recall' in logs:
                    logger.info(f"Epoch {epoch + 1}: Precision = {logs['val_precision']:.4f}, Recall = {logs['val_recall']:.4f}")
    
    def train(self, images: np.ndarray, labels: np.ndarray, 
             validation_data: Tuple[np.ndarray, np.ndarray] = None,
             model_save_path: str = None) -> keras.callbacks.History:
        """
        Enhanced training with maize-specific optimizations.
        
        Args:
            images: Training images
            labels: Training labels
            validation_data: Validation data tuple (optional)
            model_save_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()
        
        # Prepare data
        if validation_data is None:
            # Split data for validation
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, 
                test_size=self.config['validation_split'],
                stratify=np.argmax(labels, axis=1),
                random_state=42
            )
            validation_data = (val_images, val_labels)
        else:
            train_images, train_labels = images, labels
        
        # Handle class imbalance for maize diseases
        if self.config.get('data_balance', False):
            class_weights = self._calculate_class_weights(train_labels)
            self.config['class_weights'] = class_weights
            logger.info("Class weights calculated for balanced training")
        
        # Data augmentation with maize-specific strategy
        if self.config['data_augmentation']:
            train_images, train_labels = self._apply_data_augmentation(
                train_images, train_labels
            )
        
        # Get callbacks
        callbacks_list = self._get_callbacks(model_save_path)
        
        # Train model
        self.history = self.model.fit(
            train_images,
            train_labels,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks_list,
            class_weight=self.config['class_weights'],
            verbose=1
        )
        
        # Fine-tuning if enabled
        if self.config.get('fine_tuning', False):
            self._fine_tune_model(train_images, train_labels, validation_data, callbacks_list)
        
        return self.history
    
    def _calculate_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Calculate class weights to handle imbalance in maize disease datasets."""
        class_counts = np.sum(labels, axis=0)
        total_samples = len(labels)
        
        class_weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    def _fine_tune_model(self, train_images: np.ndarray, train_labels: np.ndarray,
                        validation_data: Tuple[np.ndarray, np.ndarray],
                        callbacks_list: List[keras.callbacks.Callback]):
        """Fine-tune the model by unfreezing some base layers."""
        if not hasattr(self.model, 'layers') or len(self.model.layers) < 2:
            return
        
        # Unfreeze some layers for fine-tuning
        base_model = self.model.layers[0]
        if hasattr(base_model, 'layers'):
            # Unfreeze last few layers of base model
            fine_tune_at = max(0, len(base_model.layers) - 20)
            base_model.trainable = True
            
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss=self._get_loss_function(),
                metrics=self._get_metrics()
            )
            
            logger.info(f"Fine-tuning {len(base_model.layers) - fine_tune_at} layers")
            
            # Continue training with fine-tuning
            self.model.fit(
                train_images,
                train_labels,
                batch_size=self.config['batch_size'] // 2,  # Smaller batch size
                epochs=20,  # Fewer epochs for fine-tuning
                validation_data=validation_data,
                callbacks=callbacks_list,
                class_weight=self.config['class_weights'],
                verbose=1
            )
    
    def _apply_data_augmentation(self, images: np.ndarray, 
                               labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply enhanced data augmentation for maize disease detection."""
        augmented_images = []
        augmented_labels = []
        
        strategy = self.config.get('augmentation_strategy', 'moderate')
        
        for image, label in zip(images, labels):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Determine augmentation count based on strategy
            if strategy == 'aggressive':
                aug_count = 4  # More augmentation for small datasets
            elif strategy == 'moderate':
                aug_count = 2
            else:
                aug_count = 1
            
            # Add augmented versions
            for _ in range(aug_count):
                augmented = self.data_augmentation.augment_image(image, 'training')
                augmented_images.append(augmented)
                augmented_labels.append(label)
        
        logger.info(f"Data augmentation applied: {len(images)} -> {len(augmented_images)} images")
        return np.array(augmented_images), np.array(augmented_labels)
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Enhanced model evaluation with maize-specific metrics.
        
        Args:
            test_images: Test images
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        # Get predictions
        predictions = self.model.predict(test_images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(cm):
                tp = cm[i][i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': int(np.sum(test_labels[:, i]))
                }
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist(),
            'true_classes': true_classes.tolist(),
            'class_names': self.class_names
        }
        
        return evaluation_results
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Enhanced training history visualization."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[0, 2].plot(self.history.history['precision'], label='Training Precision')
            axes[0, 2].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[0, 2].set_title('Model Precision')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 0].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 0].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 0].set_title('Model Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # F1 Score
        if 'f1_score' in self.history.history:
            axes[1, 1].plot(self.history.history['f1_score'], label='Training F1 Score')
            axes[1, 1].plot(self.history.history['val_f1_score'], label='Validation F1 Score')
            axes[1, 1].set_title('Model F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 2].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], 
                            save_path: str = None) -> None:
        """Enhanced confusion matrix visualization."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - Maize Disease Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, model_path: str, config_path: str = None) -> bool:
        """
        Save the trained model and configuration.
        
        Args:
            model_path: Path to save the model
            config_path: Path to save the configuration
            
        Returns:
            True if saved successfully
        """
        try:
            # Save model
            self.model.save(model_path)
            
            # Save configuration
            if config_path:
                config_to_save = self.config.copy()
                config_to_save['class_names'] = self.class_names
                config_to_save['training_date'] = datetime.datetime.now().isoformat()
                
                with open(config_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str, config_path: str = None) -> bool:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            config_path: Path to the configuration file
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load configuration
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.config.update(config)
                    self.class_names = config.get('class_names', [])
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get a detailed summary of the model architecture."""
        if self.model is None:
            return "No model available"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def export_model_for_mobile(self, export_path: str) -> bool:
        """Export model for mobile deployment (TensorFlow Lite)."""
        try:
            if self.model is None:
                raise ValueError("No model available for export")
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model exported to TensorFlow Lite: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return False 
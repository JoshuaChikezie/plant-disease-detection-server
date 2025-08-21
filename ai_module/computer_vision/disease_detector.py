"""
Disease Detector - Core CNN-based plant disease classification

This module provides the main interface for detecting plant diseases
using convolutional neural networks trained on Ghanaian crop datasets.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor
from .data_augmentation import DataAugmentation

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """
    Main class for plant disease detection using CNN models.
    
    Supports multiple crops common in Ghana:
    - Cassava (Manihot esculenta)
    - Maize (Zea mays) 
    - Cocoa (Theobroma cacao)
    - Yam (Dioscorea spp.)
    - Plantain (Musa spp.)
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize the disease detector.
        
        Args:
            model_path: Path to the trained CNN model
            config_path: Path to model configuration file
        """
        self.model = None
        self.image_processor = ImageProcessor()
        self.data_augmentation = DataAugmentation()
        
        # Enhanced disease classes for Ghanaian crops with maize focus
        self.disease_classes = {
            'cassava': {
                'healthy': 'Healthy Cassava',
                'cassava_mosaic_disease': 'Cassava Mosaic Disease',
                'cassava_brown_streak': 'Cassava Brown Streak Disease',
                'cassava_bacterial_blight': 'Cassava Bacterial Blight'
            },
            'maize': {
                'healthy': 'Healthy Maize',
                'maize_rust': 'Maize Rust (Puccinia sorghi)',
                'maize_smut': 'Maize Smut (Ustilago maydis)',
                'maize_leaf_blight': 'Northern Corn Leaf Blight (Exserohilum turcicum)',
                'maize_gray_leaf_spot': 'Gray Leaf Spot (Cercospora zeae-maydis)',
                'maize_common_rust': 'Common Rust (Puccinia sorghi)',
                'maize_southern_leaf_blight': 'Southern Leaf Blight (Bipolaris maydis)',
                'maize_anthracnose': 'Anthracnose Leaf Blight (Colletotrichum graminicola)',
                'maize_eyespot': 'Eyespot (Kabatiella zeae)',
                'maize_stewart_wilt': 'Stewart\'s Wilt (Pantoea stewartii)'
            },
            'cocoa': {
                'healthy': 'Healthy Cocoa',
                'cocoa_black_pod': 'Cocoa Black Pod Disease',
                'cocoa_swollen_shoot': 'Cocoa Swollen Shoot Virus',
                'cocoa_mirid_bug_damage': 'Cocoa Mirid Bug Damage'
            }
        }
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the disease detector."""
        return {
            'input_shape': (224, 224, 3),
            'confidence_threshold': 0.7,
            'max_predictions': 5,
            'preprocessing': {
                'normalize': True,
                'resize': True,
                'augment': False,
                'enhance_contrast': True,
                'remove_noise': True
            },
            'model_architecture': 'efficientnet_b0',
            'supported_crops': ['cassava', 'maize', 'cocoa'],
            'maize_specific': True,  # Flag for maize-focused detection
            'ensemble_models': False,  # Use multiple models for better accuracy
            'post_processing': {
                'confidence_calibration': True,
                'disease_stage_estimation': True,
                'severity_assessment': True
            }
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained CNN model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Try loading with custom objects for custom layers
            custom_objects = {
                'F1Score': self._f1_score_metric,
                'Precision': tf.keras.metrics.Precision,
                'Recall': tf.keras.metrics.Recall
            }
            
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def _f1_score_metric(self, y_true, y_pred):
        """Custom F1 score metric for model evaluation."""
        precision = tf.keras.metrics.Precision()(y_true, y_pred)
        recall = tf.keras.metrics.Recall()(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    def load_config(self, config_path: str) -> bool:
        """
        Load model configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            bool: True if config loaded successfully
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for better model performance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
        """
        # Apply noise reduction if enabled
        if self.config['preprocessing']['remove_noise']:
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Enhance contrast if enabled
        if self.config['preprocessing']['enhance_contrast']:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Resize image
        if self.config['preprocessing']['resize']:
            target_size = self.config['input_shape'][:2]
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        if self.config['preprocessing']['normalize']:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def detect_disease(self, image_path: str, crop_type: str = None) -> Dict[str, Any]:
        """
        Enhanced disease detection with maize-specific optimizations.
        
        Args:
            image_path: Path to the input image
            crop_type: Type of crop (optional, will auto-detect if not provided)
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Load and preprocess image
            image = self.image_processor.load_image(image_path)
            processed_image = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Process results with enhanced post-processing
            results = self._process_predictions(predictions[0], crop_type)
            
            # Add metadata and enhanced information
            results['image_path'] = image_path
            results['crop_type'] = crop_type or 'auto_detected'
            results['confidence_threshold'] = self.config['confidence_threshold']
            results['model_architecture'] = self.config['model_architecture']
            results['processing_timestamp'] = tf.timestamp().numpy().item()
            
            # Add severity assessment if enabled
            if self.config['post_processing']['severity_assessment']:
                results['severity_assessment'] = self._assess_disease_severity(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during disease detection: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _process_predictions(self, predictions: np.ndarray, crop_type: str) -> Dict[str, Any]:
        """
        Enhanced prediction processing with confidence calibration.
        
        Args:
            predictions: Raw model predictions
            crop_type: Type of crop
            
        Returns:
            Formatted prediction results
        """
        # Apply confidence calibration if enabled
        if self.config['post_processing']['confidence_calibration']:
            predictions = self._calibrate_confidence(predictions)
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:self.config['max_predictions']]
        
        results = {
            'success': True,
            'predictions': [],
            'primary_diagnosis': None,
            'confidence_score': 0.0,
            'disease_stage': 'unknown',
            'disease_confidence': 'low'
        }
        
        # Determine crop type if not provided
        if not crop_type:
            crop_type = self._auto_detect_crop(predictions)
        
        # Get disease classes for the crop
        crop_diseases = self.disease_classes.get(crop_type, {})
        
        for i, idx in enumerate(top_indices):
            confidence = float(predictions[idx])
            
            if confidence >= self.config['confidence_threshold']:
                # Map index to disease name
                disease_name = list(crop_diseases.keys())[idx] if idx < len(crop_diseases) else f"disease_{idx}"
                disease_label = crop_diseases.get(disease_name, disease_name)
                
                prediction = {
                    'disease': disease_name,
                    'label': disease_label,
                    'confidence': confidence,
                    'rank': i + 1,
                    'severity_level': self._estimate_severity_level(confidence),
                    'treatment_urgency': self._assess_treatment_urgency(confidence, disease_name)
                }
                
                results['predictions'].append(prediction)
                
                # Set primary diagnosis
                if i == 0:
                    results['primary_diagnosis'] = disease_name
                    results['confidence_score'] = confidence
                    results['disease_stage'] = self._estimate_disease_stage(confidence)
                    results['disease_confidence'] = self._get_confidence_level(confidence)
        
        return results
    
    def _calibrate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling for better confidence calibration."""
        try:
            # Simple temperature scaling
            temperature = 1.5
            calibrated = predictions / temperature
            calibrated = tf.nn.softmax(calibrated).numpy()
            return calibrated
        except:
            return predictions
    
    def _estimate_severity_level(self, confidence: float) -> str:
        """Estimate disease severity based on confidence score."""
        if confidence >= 0.95:
            return 'critical'
        elif confidence >= 0.85:
            return 'severe'
        elif confidence >= 0.75:
            return 'moderate'
        elif confidence >= 0.65:
            return 'mild'
        else:
            return 'early'
    
    def _assess_treatment_urgency(self, confidence: float, disease_name: str) -> str:
        """Assess treatment urgency based on disease and confidence."""
        high_urgency_diseases = ['maize_rust', 'maize_smut', 'cassava_mosaic_disease']
        
        if disease_name in high_urgency_diseases and confidence >= 0.7:
            return 'immediate'
        elif confidence >= 0.8:
            return 'urgent'
        elif confidence >= 0.6:
            return 'moderate'
        else:
            return 'low'
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.8:
            return 'high'
        elif confidence >= 0.7:
            return 'moderate'
        elif confidence >= 0.6:
            return 'low'
        else:
            return 'very_low'
    
    def _auto_detect_crop(self, predictions: np.ndarray) -> str:
        """
        Enhanced auto-detection of crop type based on prediction patterns.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Detected crop type
        """
        # For maize-specific models, prioritize maize detection
        if self.config.get('maize_specific', False):
            return 'maize'
        
        # Simple heuristic: assume the crop with highest overall confidence
        # In a real implementation, this would use a separate crop classifier
        return 'maize'  # Default to maize for now
    
    def _estimate_disease_stage(self, confidence: float) -> str:
        """
        Enhanced disease stage estimation.
        
        Args:
            confidence: Model confidence score
            
        Returns:
            Estimated disease stage
        """
        if confidence >= 0.95:
            return 'advanced'
        elif confidence >= 0.85:
            return 'severe'
        elif confidence >= 0.75:
            return 'moderate'
        elif confidence >= 0.65:
            return 'early'
        else:
            return 'very_early'
    
    def _assess_disease_severity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall disease severity and provide recommendations."""
        if not results.get('predictions'):
            return {'overall_severity': 'unknown', 'recommendations': []}
        
        # Calculate overall severity
        confidences = [pred['confidence'] for pred in results['predictions']]
        avg_confidence = np.mean(confidences)
        max_confidence = max(confidences)
        
        if max_confidence >= 0.9:
            overall_severity = 'critical'
        elif max_confidence >= 0.8:
            overall_severity = 'severe'
        elif max_confidence >= 0.7:
            overall_severity = 'moderate'
        else:
            overall_severity = 'mild'
        
        # Generate severity-based recommendations
        recommendations = []
        if overall_severity in ['critical', 'severe']:
            recommendations.extend([
                'Immediate intervention required',
                'Consider professional consultation',
                'Implement quarantine measures'
            ])
        elif overall_severity == 'moderate':
            recommendations.extend([
                'Monitor closely',
                'Begin treatment protocol',
                'Document progression'
            ])
        else:
            recommendations.extend([
                'Continue monitoring',
                'Implement preventive measures',
                'Maintain good agricultural practices'
            ])
        
        return {
            'overall_severity': overall_severity,
            'average_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'recommendations': recommendations
        }
    
    def batch_detect(self, image_paths: List[str], crop_type: str = None) -> List[Dict[str, Any]]:
        """
        Perform disease detection on multiple images.
        
        Args:
            image_paths: List of image paths
            crop_type: Type of crop
            
        Returns:
            List of detection results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect_disease(image_path, crop_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def get_disease_info(self, disease_name: str, crop_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific disease.
        
        Args:
            disease_name: Name of the disease
            crop_type: Type of crop
            
        Returns:
            Disease information dictionary
        """
        # Enhanced disease information for maize
        maize_disease_info = {
            'maize_rust': {
                'symptoms': ['Orange to brown pustules on leaves', 'Reduced photosynthesis', 'Premature leaf death'],
                'causes': ['Fungal pathogen Puccinia sorghi', 'High humidity', 'Warm temperatures'],
                'treatments': ['Fungicide application', 'Resistant varieties', 'Crop rotation'],
                'prevention': ['Plant resistant varieties', 'Proper spacing', 'Avoid overhead irrigation']
            },
            'maize_smut': {
                'symptoms': ['Swollen galls on ears', 'Black spore masses', 'Distorted plant parts'],
                'causes': ['Fungal pathogen Ustilago maydis', 'Wound infection', 'High nitrogen'],
                'treatments': ['Remove infected plants', 'Fungicide seed treatment', 'Crop rotation'],
                'prevention': ['Use treated seeds', 'Avoid mechanical damage', 'Balanced fertilization']
            },
            'maize_leaf_blight': {
                'symptoms': ['Elliptical lesions on leaves', 'Gray to tan centers', 'Brown borders'],
                'causes': ['Fungal pathogen Exserohilum turcicum', 'Wet conditions', 'Residue-borne'],
                'treatments': ['Fungicide application', 'Residue management', 'Resistant varieties'],
                'prevention': ['Tillage practices', 'Crop rotation', 'Resistant hybrids']
            }
        }
        
        # Return maize-specific info if available
        if crop_type == 'maize' and disease_name in maize_disease_info:
            return maize_disease_info[disease_name]
        
        # Default disease information
        disease_info = {
            'name': disease_name,
            'crop': crop_type,
            'symptoms': [],
            'causes': [],
            'treatments': [],
            'prevention': [],
            'lifecycle': {}
        }
        
        return disease_info
    
    def get_crop_diseases(self, crop_type: str) -> List[Dict[str, Any]]:
        """Get all diseases for a specific crop."""
        if crop_type not in self.disease_classes:
            return []
        
        diseases = []
        for disease_key, disease_label in self.disease_classes[crop_type].items():
            diseases.append({
                'key': disease_key,
                'label': disease_label,
                'crop': crop_type
            })
        
        return diseases
    
    def visualize_detection(self, image_path: str, results: Dict[str, Any], 
                          output_path: str = None) -> np.ndarray:
        """
        Create a visualization of the detection results.
        
        Args:
            image_path: Path to the original image
            results: Detection results
            output_path: Path to save visualization (optional)
            
        Returns:
            Visualization image as numpy array
        """
        # Load original image
        image = self.image_processor.load_image(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Results visualization
        if results['success'] and results['predictions']:
            # Create bar chart of predictions
            diseases = [pred['label'] for pred in results['predictions']]
            confidences = [pred['confidence'] for pred in results['predictions']]
            
            bars = ax2.barh(diseases, confidences, color='skyblue')
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Disease Predictions')
            ax2.set_xlim(0, 1)
            
            # Add confidence values on bars
            for bar, conf in zip(bars, confidences):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{conf:.3f}', va='center')
        else:
            ax2.text(0.5, 0.5, 'No predictions available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Detection Results')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': type(self.model).__name__,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights]),
            'config': self.config,
            'maize_specific': self.config.get('maize_specific', False),
            'supported_crops': self.config.get('supported_crops', [])
        } 
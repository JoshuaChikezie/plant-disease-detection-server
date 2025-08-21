"""
Enhanced Data Augmentation for Plant Disease Detection

This module provides comprehensive data augmentation techniques specifically
optimized for plant disease detection, with maize-specific strategies.
"""

import cv2
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imgaug.augmenters as iaa
from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes
import logging

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Enhanced data augmentation class for plant disease detection.
    
    Provides multiple augmentation strategies:
    - Basic augmentation for general use
    - Disease-specific augmentation for training
    - Maize-specific augmentation strategies
    - Validation augmentation for inference
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data augmentation module.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or self._get_default_config()
        self._setup_augmenters()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default augmentation configuration."""
        return {
            'maize_specific': True,
            'disease_focused': True,
            'preserve_disease_features': True,
            'augmentation_intensity': 'moderate',  # light, moderate, aggressive
            'color_augmentation': True,
            'geometric_augmentation': True,
            'noise_augmentation': True,
            'weather_simulation': True,
            'lighting_variations': True,
            'perspective_changes': True,
            'blur_and_sharpness': True,
            'elastic_deformation': True,
            'morphological_operations': True
        }
    
    def _setup_augmenters(self):
        """Setup different augmentation pipelines."""
        # Basic augmentation pipeline
        self.basic_augmenter = self._create_basic_pipeline()
        
        # Disease-focused augmentation pipeline
        self.disease_augmenter = self._create_disease_focused_pipeline()
        
        # Maize-specific augmentation pipeline
        self.maize_augmenter = self._create_maize_specific_pipeline()
        
        # Validation pipeline (minimal augmentation)
        self.validation_augmenter = self._create_validation_pipeline()
        
        # Albumentations pipeline for advanced augmentations
        self.albumentations_pipeline = self._create_albumentations_pipeline()
    
    def _create_basic_pipeline(self) -> Sequential:
        """Create basic augmentation pipeline."""
        return Sequential([
            # Geometric transformations
            Sometimes(0.5, iaa.Affine(
                rotate=(-15, 15),
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            )),
            
            # Color adjustments
            Sometimes(0.3, iaa.Multiply((0.8, 1.2))),
            Sometimes(0.3, iaa.ContrastNormalization((0.8, 1.2))),
            
            # Noise and blur
            Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)))
        ])
    
    def _create_disease_focused_pipeline(self) -> Sequential:
        """Create disease-focused augmentation pipeline that preserves disease features."""
        return Sequential([
            # Geometric transformations (preserve disease patterns)
            Sometimes(0.6, iaa.Affine(
                rotate=(-20, 20),
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                shear=(-8, 8)
            )),
            
            # Color variations (important for disease detection)
            Sometimes(0.4, iaa.Multiply((0.7, 1.3))),
            Sometimes(0.4, iaa.ContrastNormalization((0.7, 1.3))),
            Sometimes(0.3, iaa.AddToHue((-20, 20))),
            Sometimes(0.3, iaa.AddToSaturation((-20, 20))),
            
            # Lighting variations (simulate different field conditions)
            Sometimes(0.3, iaa.Add((-30, 30))),
            Sometimes(0.2, iaa.Multiply((0.6, 1.4))),
            
            # Noise and blur (realistic field conditions)
            Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.8))),
            Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255))),
            
            # Weather simulation
            Sometimes(0.1, iaa.Rain(speed=(0.1, 0.3))),
            Sometimes(0.05, iaa.Fog())
        ])
    
    def _create_maize_specific_pipeline(self) -> Sequential:
        """Create maize-specific augmentation pipeline."""
        return Sequential([
            # Maize-specific geometric transformations
            Sometimes(0.7, iaa.Affine(
                rotate=(-25, 25),  # Maize leaves can be at various angles
                scale=(0.8, 1.2),   # Different distances from camera
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                shear=(-10, 10)     # Leaf bending
            )),
            
            # Maize-specific color variations
            Sometimes(0.5, iaa.Multiply((0.6, 1.4))),  # Different lighting conditions
            Sometimes(0.4, iaa.ContrastNormalization((0.6, 1.4))),
            Sometimes(0.4, iaa.AddToHue((-25, 25))),   # Color variations in maize
            Sometimes(0.4, iaa.AddToSaturation((-25, 25))),
            Sometimes(0.3, iaa.AddToBrightness((-40, 40))),
            
            # Field condition simulation
            Sometimes(0.3, iaa.Add((-40, 40))),         # Different exposure
            Sometimes(0.2, iaa.Multiply((0.5, 1.5))),   # Brightness variations
            
            # Maize-specific noise and blur
            Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),  # Wind movement
            Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
            Sometimes(0.1, iaa.SaltAndPepper(0.01)),           # Dust particles
            
            # Weather effects common in maize fields
            Sometimes(0.15, iaa.Rain(speed=(0.1, 0.4))),       # Rain
            Sometimes(0.1, iaa.Fog()),                         # Morning fog
            Sometimes(0.05, iaa.Snowflakes(flake_size=(0.1, 0.3))),  # Early frost
            
            # Perspective changes (different camera angles)
            Sometimes(0.2, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            
            # Elastic deformation (leaf movement)
            Sometimes(0.1, iaa.ElasticTransformation(alpha=(0, 50), sigma=5))
        ])
    
    def _create_validation_pipeline(self) -> Sequential:
        """Create minimal augmentation pipeline for validation."""
        return Sequential([
            # Only basic resizing and normalization
            iaa.Resize({"height": 224, "width": 224}),
            iaa.Normalize()
        ])
    
    def _create_albumentations_pipeline(self) -> A.Compose:
        """Create Albumentations pipeline for advanced augmentations."""
        return A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            
            # Color and contrast
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            
            # Normalization
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'basic') -> np.ndarray:
        """
        Apply augmentation to an image.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation ('basic', 'disease', 'maize', 'validation')
            
        Returns:
            Augmented image
        """
        try:
            if augmentation_type == 'basic':
                augmented = self.basic_augmenter.augment_image(image)
            elif augmentation_type == 'disease':
                augmented = self.disease_augmenter.augment_image(image)
            elif augmentation_type == 'maize':
                augmented = self.maize_augmenter.augment_image(image)
            elif augmentation_type == 'validation':
                augmented = self.validation_augmenter.augment_image(image)
            elif augmentation_type == 'training':
                # Use maize-specific for training if configured
                if self.config.get('maize_specific', False):
                    augmented = self.maize_augmenter.augment_image(image)
                else:
                    augmented = self.disease_augmenter.augment_image(image)
            else:
                augmented = self.basic_augmenter.augment_image(image)
            
            # Ensure image is in valid range
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            return image
    
    def augment_batch(self, images: np.ndarray, augmentation_type: str = 'basic') -> np.ndarray:
        """
        Apply augmentation to a batch of images.
        
        Args:
            images: Batch of input images
            augmentation_type: Type of augmentation
            
        Returns:
            Batch of augmented images
        """
        try:
            if augmentation_type == 'basic':
                augmented = self.basic_augmenter.augment_images(images)
            elif augmentation_type == 'disease':
                augmented = self.disease_augmenter.augment_images(images)
            elif augmentation_type == 'maize':
                augmented = self.maize_augmenter.augment_images(images)
            elif augmentation_type == 'validation':
                augmented = self.validation_augmenter.augment_images(images)
            else:
                augmented = self.basic_augmenter.augment_images(images)
            
            # Ensure images are in valid range
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Batch augmentation failed: {e}")
            return images
    
    def apply_maize_specific_augmentation(self, image: np.ndarray, 
                                        intensity: str = 'moderate') -> np.ndarray:
        """
        Apply maize-specific augmentation with controlled intensity.
        
        Args:
            image: Input image
            intensity: Augmentation intensity ('light', 'moderate', 'aggressive')
            
        Returns:
            Augmented image
        """
        if not self.config.get('maize_specific', False):
            return self.augment_image(image, 'basic')
        
        # Create intensity-specific pipeline
        if intensity == 'light':
            pipeline = self._create_light_maize_pipeline()
        elif intensity == 'aggressive':
            pipeline = self._create_aggressive_maize_pipeline()
        else:
            pipeline = self.maize_augmenter  # Default moderate
        
        try:
            augmented = pipeline.augment_image(image)
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
            return augmented
        except Exception as e:
            logger.error(f"Maize-specific augmentation failed: {e}")
            return image
    
    def _create_light_maize_pipeline(self) -> Sequential:
        """Create light maize augmentation pipeline."""
        return Sequential([
            # Minimal geometric transformations
            Sometimes(0.3, iaa.Affine(
                rotate=(-10, 10),
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            )),
            
            # Light color variations
            Sometimes(0.2, iaa.Multiply((0.8, 1.2))),
            Sometimes(0.2, iaa.ContrastNormalization((0.8, 1.2))),
            
            # Minimal noise
            Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 0.3)))
        ])
    
    def _create_aggressive_maize_pipeline(self) -> Sequential:
        """Create aggressive maize augmentation pipeline."""
        return Sequential([
            # Strong geometric transformations
            Sometimes(0.8, iaa.Affine(
                rotate=(-35, 35),
                scale=(0.7, 1.3),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                shear=(-15, 15)
            )),
            
            # Strong color variations
            Sometimes(0.6, iaa.Multiply((0.5, 1.5))),
            Sometimes(0.5, iaa.ContrastNormalization((0.5, 1.5))),
            Sometimes(0.5, iaa.AddToHue((-30, 30))),
            Sometimes(0.4, iaa.AddToSaturation((-30, 30))),
            Sometimes(0.4, iaa.AddToBrightness((-50, 50))),
            
            # Strong lighting variations
            Sometimes(0.4, iaa.Add((-50, 50))),
            Sometimes(0.3, iaa.Multiply((0.4, 1.6))),
            
            # Strong noise and blur
            Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 1.5))),
            Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.06 * 255))),
            Sometimes(0.2, iaa.SaltAndPepper(0.02)),
            
            # Weather effects
            Sometimes(0.2, iaa.Rain(speed=(0.2, 0.5))),
            Sometimes(0.15, iaa.Fog()),
            Sometimes(0.1, iaa.Snowflakes(flake_size=(0.2, 0.4))),
            
            # Perspective and deformation
            Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.02, 0.15))),
            Sometimes(0.2, iaa.ElasticTransformation(alpha=(0, 70), sigma=7))
        ])
    
    def apply_disease_preserving_augmentation(self, image: np.ndarray, 
                                            disease_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply augmentation while preserving disease features.
        
        Args:
            image: Input image
            disease_mask: Optional mask of disease regions
            
        Returns:
            Augmented image with preserved disease features
        """
        if not self.config.get('preserve_disease_features', False):
            return self.augment_image(image, 'disease')
        
        try:
            # Apply disease-focused augmentation
            augmented = self.disease_augmenter.augment_image(image)
            
            # If disease mask is provided, ensure disease regions are preserved
            if disease_mask is not None:
                # Blend original disease regions with augmented image
                augmented = self._preserve_disease_regions(image, augmented, disease_mask)
            
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
            return augmented
            
        except Exception as e:
            logger.error(f"Disease-preserving augmentation failed: {e}")
            return image
    
    def _preserve_disease_regions(self, original: np.ndarray, 
                                augmented: np.ndarray, 
                                disease_mask: np.ndarray) -> np.ndarray:
        """Preserve disease regions during augmentation."""
        try:
            # Normalize mask
            mask = disease_mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1) if len(mask.shape) == 2 else mask
            
            # Blend images
            preserved = original * mask + augmented * (1 - mask)
            return preserved.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Failed to preserve disease regions: {e}")
            return augmented
    
    def create_augmentation_pipeline(self, 
                                   augmentation_type: str = 'maize',
                                   intensity: str = 'moderate',
                                   preserve_features: bool = True) -> Any:
        """
        Create a custom augmentation pipeline.
        
        Args:
            augmentation_type: Type of augmentation
            intensity: Augmentation intensity
            preserve_features: Whether to preserve disease features
            
        Returns:
            Augmentation pipeline
        """
        if augmentation_type == 'maize':
            if intensity == 'light':
                return self._create_light_maize_pipeline()
            elif intensity == 'aggressive':
                return self._create_aggressive_maize_pipeline()
            else:
                return self.maize_augmenter
        elif augmentation_type == 'disease':
            return self.disease_augmenter
        elif augmentation_type == 'basic':
            return self.basic_augmenter
        else:
            return self.basic_augmenter
    
    def get_augmentation_stats(self, image: np.ndarray, 
                             num_samples: int = 10) -> Dict[str, Any]:
        """
        Get statistics about augmentation effects.
        
        Args:
            image: Input image
            num_samples: Number of augmentation samples
            
        Returns:
            Augmentation statistics
        """
        try:
            stats = {
                'original_shape': image.shape,
                'augmentation_samples': []
            }
            
            # Generate multiple augmented samples
            for i in range(num_samples):
                augmented = self.augment_image(image, 'maize')
                stats['augmentation_samples'].append({
                    'sample_id': i,
                    'shape': augmented.shape,
                    'mean_color': np.mean(augmented, axis=(0, 1)).tolist(),
                    'std_color': np.std(augmented, axis=(0, 1)).tolist()
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get augmentation stats: {e}")
            return {'error': str(e)}
    
    def validate_augmentation(self, image: np.ndarray, 
                            augmentation_type: str = 'maize') -> bool:
        """
        Validate that augmentation produces valid results.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation
            
        Returns:
            True if augmentation is valid
        """
        try:
            # Apply augmentation
            augmented = self.augment_image(image, augmentation_type)
            
            # Check basic properties
            if augmented.shape != image.shape:
                return False
            
            if not np.isfinite(augmented).all():
                return False
            
            if augmented.min() < 0 or augmented.max() > 255:
                return False
            
            # Check that image is not completely black or white
            if np.mean(augmented) < 10 or np.mean(augmented) > 245:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Augmentation validation failed: {e}")
            return False 
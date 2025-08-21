"""
Computer Vision Module for Plant Disease Detection

This module handles:
- Image preprocessing and augmentation
- CNN model training and inference
- Disease classification and confidence scoring
- Progressive disease stage tracking
- Multi-crop support for Ghanaian agriculture
"""

from .disease_detector import DiseaseDetector
from .image_processor import ImageProcessor
from .model_trainer import ModelTrainer
from .data_augmentation import DataAugmentation

__all__ = [
    "DiseaseDetector",
    "ImageProcessor",
    "ModelTrainer", 
    "DataAugmentation"
] 
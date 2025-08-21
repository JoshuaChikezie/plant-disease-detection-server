"""
Image Processor - Handles image loading, preprocessing, and utility functions

This module provides comprehensive image processing capabilities for
plant disease detection, including loading, resizing, filtering, and
quality assessment.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Comprehensive image processing class for plant disease detection.
    
    Provides methods for:
    - Image loading and validation
    - Preprocessing and enhancement
    - Quality assessment
    - Format conversion
    - Batch processing
    """
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.max_image_size = (4096, 4096)  # Maximum dimensions
        self.min_image_size = (64, 64)      # Minimum dimensions
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array in BGR format
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {file_ext}")
        
        try:
            # Load image using OpenCV (BGR format)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def load_image_pil(self, image_path: str) -> Image.Image:
        """
        Load an image using PIL (Python Imaging Library).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            logger.error(f"Error loading image with PIL {image_path}: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = 'bilinear') -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image as numpy array
            target_size: Target (width, height)
            method: Interpolation method ('bilinear', 'bicubic', 'nearest')
            
        Returns:
            Resized image
        """
        interpolation_map = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interpolation = interpolation_map.get(method, cv2.INTER_LINEAR)
        resized = cv2.resize(image, target_size, interpolation=interpolation)
        
        return resized
    
    def crop_image(self, image: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image to specified region.
        
        Args:
            image: Input image
            crop_box: (x, y, width, height) crop region
            
        Returns:
            Cropped image
        """
        x, y, w, h = crop_box
        return image[y:y+h, x:x+w]
    
    def center_crop(self, image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Center crop image to specified size.
        
        Args:
            image: Input image
            crop_size: (width, height) of crop
            
        Returns:
            Center-cropped image
        """
        h, w = image.shape[:2]
        crop_w, crop_h = crop_size
        
        # Calculate crop coordinates
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        
        return self.crop_image(image, (x, y, crop_w, crop_h))
    
    def normalize_image(self, image: np.ndarray, mean: List[float] = None, 
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
            mean: Mean values for each channel (BGR)
            std: Standard deviation values for each channel (BGR)
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet means (RGB)
        if std is None:
            std = [0.229, 0.224, 0.225]   # ImageNet stds (RGB)
        
        # Convert BGR to RGB for normalization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = image_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        for i in range(3):
            normalized[:, :, i] = (normalized[:, :, i] - mean[i]) / std[i]
        
        return normalized
    
    def enhance_image(self, image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Enhance image quality using various filters.
        
        Args:
            image: Input image
            brightness: Brightness factor (0.0 to 2.0)
            contrast: Contrast factor (0.0 to 2.0)
            saturation: Saturation factor (0.0 to 2.0)
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(saturation)
        
        # Convert back to numpy array
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def apply_filters(self, image: np.ndarray, filter_type: str = 'gaussian', 
                     kernel_size: int = 3) -> np.ndarray:
        """
        Apply various image filters.
        
        Args:
            image: Input image
            filter_type: Type of filter ('gaussian', 'median', 'bilateral')
            kernel_size: Size of filter kernel
            
        Returns:
            Filtered image
        """
        if filter_type == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif filter_type == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            return image
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny', 
                    threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
        """
        Detect edges in the image.
        
        Args:
            image: Input image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            threshold1: First threshold for Canny
            threshold2: Second threshold for Canny
            
        Returns:
            Edge map
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == 'canny':
            return cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2)
        elif method == 'laplacian':
            return cv2.Laplacian(gray, cv2.CV_64F)
        else:
            return gray
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess image quality for disease detection.
        
        Args:
            image: Input image
            
        Returns:
            Quality assessment dictionary
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        height, width = gray.shape
        total_pixels = height * width
        
        # Brightness
        mean_brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Noise estimation
        noise = np.mean(np.abs(cv2.GaussianBlur(gray, (3, 3), 0) - gray))
        
        # Blur detection
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        blur_score = np.mean(magnitude_spectrum)
        
        # Quality scores
        brightness_score = 1.0 if 50 <= mean_brightness <= 200 else 0.5
        contrast_score = 1.0 if contrast > 30 else 0.5
        sharpness_score = 1.0 if sharpness > 100 else 0.5
        noise_score = 1.0 if noise < 10 else 0.5
        
        overall_score = (brightness_score + contrast_score + 
                        sharpness_score + noise_score) / 4
        
        return {
            'dimensions': (width, height),
            'total_pixels': total_pixels,
            'mean_brightness': float(mean_brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'noise_level': float(noise),
            'blur_score': float(blur_score),
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'sharpness_score': sharpness_score,
            'noise_score': noise_score,
            'overall_quality_score': overall_score,
            'is_suitable_for_detection': overall_score > 0.7
        }
    
    def segment_plant(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment plant from background using color-based segmentation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (segmented_image, mask)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range for plants
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented, mask
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract basic image features for analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of extracted features
        """
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = {}
        
        # Color features
        features['mean_r'] = np.mean(image[:, :, 2])
        features['mean_g'] = np.mean(image[:, :, 1])
        features['mean_b'] = np.mean(image[:, :, 0])
        features['mean_h'] = np.mean(hsv[:, :, 0])
        features['mean_s'] = np.mean(hsv[:, :, 1])
        features['mean_v'] = np.mean(hsv[:, :, 2])
        
        # Texture features
        features['std_gray'] = np.std(gray)
        features['entropy'] = self._calculate_entropy(gray)
        
        # Shape features
        features['aspect_ratio'] = image.shape[1] / image.shape[0]
        features['area'] = image.shape[0] * image.shape[1]
        
        return features
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy as a measure of texture complexity."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    
    def save_image(self, image: np.ndarray, output_path: str, 
                  quality: int = 95) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (1-100)
            
        Returns:
            True if saved successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    def batch_process(self, image_paths: List[str], 
                     operations: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Apply a series of operations to multiple images.
        
        Args:
            image_paths: List of image paths
            operations: List of operation dictionaries
            
        Returns:
            List of processed images
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                image = self.load_image(image_path)
                
                # Apply operations
                for operation in operations:
                    op_type = operation.get('type')
                    params = operation.get('params', {})
                    
                    if op_type == 'resize':
                        image = self.resize_image(image, params['target_size'])
                    elif op_type == 'enhance':
                        image = self.enhance_image(image, **params)
                    elif op_type == 'filter':
                        image = self.apply_filters(image, **params)
                    elif op_type == 'normalize':
                        image = self.normalize_image(image, **params)
                
                processed_images.append(image)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                processed_images.append(None)
        
        return processed_images 
"""
Language Processor - Multilingual text processing for plant disease detection

This module provides comprehensive natural language processing capabilities
for multiple Ghanaian languages including English, Twi, Ga, Hausa, and Ewe.
Supports text preprocessing, language detection, agricultural term extraction,
sentiment analysis, and entity recognition for agricultural contexts.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

# Optional imports for advanced NLP features
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Advanced NLP features will be limited.")

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """
    Main language processor for multilingual plant disease detection.
    
    Supports languages commonly used in Ghana:
    - English (en) - Official language
    - Twi (tw) - Akan language, widely spoken
    - Ga (ga) - Ga-Adangbe language, Greater Accra
    - Hausa (ha) - Northern Ghana language
    - Ewe (ee) - Volta Region language
    
    Features:
    - Language detection and preprocessing
    - Agricultural term extraction
    - Disease indicator identification
    - Sentiment analysis for farmer feedback
    - Named entity recognition
    - Text classification for agricultural contexts
    """
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the language processor.
        
        Args:
            model_path: Path to pre-trained language models (optional)
            config_path: Path to language configuration file (optional)
        """
        # Supported languages with their names
        self.supported_languages = {
            'en': 'English',
            'tw': 'Twi',
            'ga': 'Ga', 
            'ha': 'Hausa',
            'ee': 'Ewe'
        }
        
        # Load language-specific configurations
        self.language_configs = self._load_language_configs()
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.sentiment_analyzer = None
        self.text_classifier = None
        
        # Load models if available and path provided
        if model_path and TRANSFORMERS_AVAILABLE:
            self.load_models(model_path)
        elif not TRANSFORMERS_AVAILABLE:
            logger.info("Using basic NLP features without transformers")
        
        # Load custom configuration if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def _load_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load language-specific configurations including stop words,
        agricultural terms, and disease indicators for each supported language.
        
        Returns:
            Dictionary containing language configurations
        """
        return {
            'en': {
                'stop_words': [
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
                ],
                'agricultural_terms': [
                    'plant', 'disease', 'leaf', 'stem', 'root', 'fruit', 'seed', 'soil', 
                    'water', 'fertilizer', 'pesticide', 'fungicide', 'harvest', 'crop', 
                    'farm', 'farmer', 'agriculture', 'cassava', 'maize', 'cocoa', 'yam',
                    'plantain', 'rice', 'millet', 'sorghum', 'groundnut', 'cowpea'
                ],
                'disease_indicators': [
                    'yellow', 'brown', 'black', 'spots', 'wilting', 'drying', 'rot', 'mold', 
                    'fungus', 'infection', 'sick', 'unhealthy', 'damaged', 'dead', 'blight',
                    'rust', 'mosaic', 'streak', 'curl', 'stunted', 'discolored'
                ]
            },
            'tw': {
                'stop_words': ['a', 'na', 'ne', 'wo', 'ba', 'de', 'ma', 'ye', 'no', 'so'],
                'agricultural_terms': [
                    'dua', 'yare', 'ahaban', 'nhaban', 'asase', 'nsu', 'mmera', 'aboa', 
                    'kurom', 'oboo', 'aduan', 'aduru', 'sika', 'nkate', 'bayer', 'aberewa',
                    'bankye', 'aburo', 'koko', 'bayere', 'borɔdɔ'
                ],
                'disease_indicators': [
                    'kokoo', 'tuntum', 'fitaa', 'nkyene', 'yaw', 'hye', 'kum', 'yare', 
                    'bone', 'pa', 'sɛe', 'poroo'
                ]
            },
            'ga': {
                'stop_words': ['ke', 'le', 'ye', 'wo', 'ba', 'de', 'ma', 'no', 'so', 'na'],
                'agricultural_terms': [
                    'tsɔ', 'yare', 'gbɔgbɔ', 'gbɔgbɔi', 'shikpɔŋ', 'tsui', 'nɔŋmɛi', 
                    'gbɔgbɔ', 'agble', 'nudzɔdzɔ', 'kpokplo', 'abladze'
                ],
                'disease_indicators': [
                    'kokoo', 'tuntum', 'fitaa', 'yaw', 'hye', 'kum', 'yare', 'bone', 'pa'
                ]
            },
            'ha': {
                'stop_words': ['da', 'na', 'ne', 'ya', 'ba', 'da', 'ma', 'ye', 'ba', 'da'],
                'agricultural_terms': [
                    'shuka', 'cutar', 'ganye', 'kasa', 'ruwa', 'takin', 'noma', 'manomi',
                    'abinci', 'magani', 'kudi', 'gyada', 'masara', 'wake', 'dawa', 'hatsi'
                ],
                'disease_indicators': [
                    'ja', 'baki', 'fari', 'rauni', 'zafi', 'mutu', 'cutar', 'mugun', 'kyau'
                ]
            },
            'ee': {
                'stop_words': ['la', 'na', 'ne', 'wo', 'ba', 'de', 'ma', 'ye', 'no', 'so'],
                'agricultural_terms': [
                    'nu', 'dɔ', 'aŋgba', 'anyigba', 'tsi', 'gbe', 'agble', 'agbledela',
                    'nuɖuɖu', 'atike', 'bli', 'agbeli', 'azi'
                ],
                'disease_indicators': [
                    'dzĩ', 'yibɔ', 'ɣe', 'abi', 'ku', 'dɔ', 'vɔ̃', 'gbegblẽ'
                ]
            }
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text using keyword matching.
        
        Uses language-specific agricultural terms and common words to identify
        the most likely language. Falls back to English if detection is uncertain.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (en, tw, ga, ha, ee)
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
        
        text_lower = text.lower()
        language_scores = {}
        
        # Score each language based on term matches
        for lang_code, config in self.language_configs.items():
            score = 0
            
            # Check agricultural terms (higher weight)
            for term in config.get('agricultural_terms', []):
                if term.lower() in text_lower:
                    score += 2
            
            # Check disease indicators
            for indicator in config.get('disease_indicators', []):
                if indicator.lower() in text_lower:
                    score += 1.5
            
            # Check stop words (lower weight)
            for stop_word in config.get('stop_words', []):
                if stop_word.lower() in text_lower:
                    score += 0.5
            
            language_scores[lang_code] = score
        
        # Return language with highest score, default to English
        detected_language = max(language_scores, key=language_scores.get)
        
        if language_scores[detected_language] == 0:
            detected_language = 'en'
        
        logger.debug(f"Language detected: {detected_language} (scores: {language_scores})")
        return detected_language
    
    def preprocess_text(self, text: str, language: str = None) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text
            language: Language code (optional, will auto-detect if not provided)
            
        Returns:
            Preprocessed text
        """
        if language is None:
            language = self.detect_language(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        stop_words = self.language_configs.get(language, {}).get('stop_words', [])
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def extract_agricultural_terms(self, text: str, language: str = None) -> List[str]:
        """
        Extract agricultural terms from text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of agricultural terms found
        """
        if language is None:
            language = self.detect_language(text)
        
        text_lower = text.lower()
        agricultural_terms = self.language_configs.get(language, {}).get('agricultural_terms', [])
        
        found_terms = []
        for term in agricultural_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def extract_disease_indicators(self, text: str, language: str = None) -> List[str]:
        """
        Extract disease indicators from text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of disease indicators found
        """
        if language is None:
            language = self.detect_language(text)
        
        text_lower = text.lower()
        disease_indicators = self.language_configs.get(language, {}).get('disease_indicators', [])
        
        found_indicators = []
        for indicator in disease_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return found_indicators
    
    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Sentiment analysis results
        """
        if self.sentiment_analyzer is None:
            # Initialize sentiment analyzer if not available
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception as e:
                logger.warning(f"Could not initialize sentiment analyzer: {e}")
                return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            result = self.sentiment_analyzer(text)
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score']
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def classify_text(self, text: str, categories: List[str], 
                     language: str = None) -> Dict[str, float]:
        """
        Classify text into predefined categories.
        
        Args:
            text: Input text
            categories: List of possible categories
            language: Language code
            
        Returns:
            Classification results with confidence scores
        """
        if self.text_classifier is None:
            # Initialize text classifier if not available
            try:
                self.text_classifier = pipeline("zero-shot-classification")
            except Exception as e:
                logger.warning(f"Could not initialize text classifier: {e}")
                return {category: 0.0 for category in categories}
        
        try:
            result = self.text_classifier(text, categories)
            return dict(zip(result['labels'], result['scores']))
        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            return {category: 0.0 for category in categories}
    
    def extract_entities(self, text: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction based on patterns
        entities = []
        
        # Extract crop names
        crop_patterns = {
            'en': r'\b(cassava|maize|cocoa|yam|plantain|rice|millet|sorghum)\b',
            'tw': r'\b(dua|nkate|bayer|aberewa|kokoo|agbele|mfonini)\b',
            'ga': r'\b(tsɔ|gbɔgbɔ|gbɔgbɔi|gbɔgbɔ|gbɔgbɔi)\b',
            'ha': r'\b(rogo|masara|koko|dankali|ayaba|shinkafa|maiwa|dawa)\b',
            'ee': r'\b(agbeli|agbledela|agbledela|agbledela|agbledela)\b'
        }
        
        if language is None:
            language = self.detect_language(text)
        
        pattern = crop_patterns.get(language, crop_patterns['en'])
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': 'CROP',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        # Extract disease names
        disease_patterns = {
            'en': r'\b(mosaic|blight|rust|smut|rot|wilt|spot|mildew)\b',
            'tw': r'\b(yare|ahaban|nhaban|yaw|hye|kum)\b',
            'ga': r'\b(yare|gbɔgbɔ|gbɔgbɔi|yaw|hye|kum)\b',
            'ha': r'\b(cutar|rauni|zafi|mutu|mugun)\b',
            'ee': r'\b(yare|yaw|xɔ|ku|vɔ)\b'
        }
        
        pattern = disease_patterns.get(language, disease_patterns['en'])
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': 'DISEASE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })
        
        return entities
    
    def tokenize_text(self, text: str, language: str = None) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of tokens
        """
        # Simple word tokenization
        # In production, use language-specific tokenizers
        
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        if language:
            stop_words = self.language_configs.get(language, {}).get('stop_words', [])
            tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def get_text_features(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Extract comprehensive text features.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary of text features
        """
        if language is None:
            language = self.detect_language(text)
        
        # Preprocess text
        preprocessed = self.preprocess_text(text, language)
        
        # Extract features
        tokens = self.tokenize_text(preprocessed, language)
        agricultural_terms = self.extract_agricultural_terms(text, language)
        disease_indicators = self.extract_disease_indicators(text, language)
        entities = self.extract_entities(text, language)
        sentiment = self.analyze_sentiment(text, language)
        
        features = {
            'language': language,
            'original_text': text,
            'preprocessed_text': preprocessed,
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'agricultural_terms': agricultural_terms,
            'disease_indicators': disease_indicators,
            'entities': entities,
            'sentiment': sentiment,
            'has_agricultural_content': len(agricultural_terms) > 0,
            'has_disease_indicators': len(disease_indicators) > 0,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return features
    
    def load_models(self, model_path: str) -> bool:
        """
        Load pre-trained language models.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if models loaded successfully
        """
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            
            logger.info(f"Language models loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load language models: {e}")
            return False
    
    def load_config(self, config_path: str) -> bool:
        """
        Load language configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if config loaded successfully
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.language_configs.update(config.get('language_configs', {}))
            
            logger.info(f"Language configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load language configuration: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save language configuration to file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            True if config saved successfully
        """
        try:
            config = {
                'language_configs': self.language_configs,
                'supported_languages': self.supported_languages
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Language configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save language configuration: {e}")
            return False 
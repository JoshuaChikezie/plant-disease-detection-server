"""
Knowledge Base Module for Plant Disease Detection

This module handles:
- Web scraping from agricultural databases and research papers
- Data collection and storage
- Disease information management
- Treatment and prevention knowledge
- Local agricultural practices
"""

from .knowledge_base import KnowledgeBase
from .web_scraper import WebScraper
from .data_manager import DataManager
from .disease_database import DiseaseDatabase

__all__ = [
    "KnowledgeBase",
    "WebScraper",
    "DataManager",
    "DiseaseDatabase"
] 
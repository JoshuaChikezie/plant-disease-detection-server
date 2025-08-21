"""
Knowledge Base - Agricultural data and disease information management

This module provides comprehensive knowledge management for plant disease
detection, including disease information, treatments, and local practices.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
import requests

from .web_scraper import WebScraper
from .data_manager import DataManager
from .disease_database import DiseaseDatabase

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Main knowledge base for plant disease detection.
    
    Manages:
    - Disease information and symptoms
    - Treatment options and recommendations
    - Local agricultural practices
    - Research data and publications
    - Farmer feedback and corrections
    """
    
    def __init__(self, db_path: str = None, config_path: str = None):
        """
        Initialize the knowledge base.
        
        Args:
            db_path: Path to knowledge base database
            config_path: Path to configuration file
        """
        self.db_path = db_path or "data/knowledge_base.db"
        self.config_path = config_path or "config/knowledge_base.json"
        
        # Initialize components
        self.web_scraper = WebScraper()
        self.data_manager = DataManager()
        self.disease_database = DiseaseDatabase()
        
        # Knowledge base structure
        self.knowledge_structure = {
            'diseases': {},
            'crops': {},
            'treatments': {},
            'practices': {},
            'research': {},
            'feedback': {}
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self._init_database()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load knowledge base configuration."""
        default_config = {
            'data_sources': {
                'agricultural_databases': [
                    'https://www.fao.org/agriculture/crops/',
                    'https://www.cabi.org/',
                    'https://www.plantwise.org/'
                ],
                'research_institutions': [
                    'https://www.csir.org.gh/',
                    'https://www.mofa.gov.gh/',
                    'https://www.ug.edu.gh/'
                ],
                'local_sources': [
                    'https://ghanafarming.com/',
                    'https://agricghana.com/'
                ]
            },
            'update_frequency': {
                'disease_data': 7,  # days
                'treatment_info': 14,  # days
                'research_papers': 30,  # days
                'local_practices': 90  # days
            },
            'supported_crops': [
                'cassava', 'maize', 'cocoa', 'yam', 'plantain', 'rice',
                'millet', 'sorghum', 'groundnut', 'cowpea'
            ],
            'supported_languages': ['en', 'tw', 'ga', 'ha', 'ee'],
            'data_retention': {
                'disease_records': 365,  # days
                'feedback_data': 730,  # days
                'research_data': 1825  # days
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def _init_database(self):
        """Initialize knowledge base database."""
        try:
            # Create database directory
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Initialize database tables
            self._create_tables()
            
            # Load initial data
            self._load_initial_data()
            
            logger.info("Knowledge base database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _create_tables(self):
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Diseases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diseases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                scientific_name TEXT,
                crop_type TEXT NOT NULL,
                symptoms TEXT,
                causes TEXT,
                lifecycle TEXT,
                severity TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Treatments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS treatments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER,
                treatment_type TEXT NOT NULL,
                description TEXT,
                application_method TEXT,
                effectiveness REAL,
                cost_category TEXT,
                availability TEXT,
                organic BOOLEAN DEFAULT FALSE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (disease_id) REFERENCES diseases (id)
            )
        ''')
        
        # Crops table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                scientific_name TEXT,
                local_names TEXT,
                growing_season TEXT,
                common_diseases TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Local practices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS local_practices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_id INTEGER,
                practice_name TEXT NOT NULL,
                description TEXT,
                region TEXT,
                effectiveness REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crop_id) REFERENCES crops (id)
            )
        ''')
        
        # Research papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                publication_date DATE,
                source TEXT,
                url TEXT,
                keywords TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Farmer feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS farmer_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER,
                farmer_id TEXT,
                feedback_type TEXT,
                feedback_text TEXT,
                confidence_score REAL,
                location TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (disease_id) REFERENCES diseases (id)
            )
        ''')
        
        # Data sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                source_url TEXT,
                source_type TEXT,
                last_updated TIMESTAMP,
                status TEXT DEFAULT 'active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_initial_data(self):
        """Load initial knowledge base data."""
        # Load Ghanaian crop data
        self._load_crop_data()
        
        # Load common disease data
        self._load_disease_data()
        
        # Load local practices
        self._load_local_practices()
    
    def _load_crop_data(self):
        """Load initial crop data for Ghanaian agriculture."""
        crops_data = [
            {
                'name': 'Cassava',
                'scientific_name': 'Manihot esculenta',
                'local_names': '{"tw": "Bankye", "ga": "Bankye", "ha": "Rogo", "ee": "Agbeli"}',
                'growing_season': 'Year-round',
                'common_diseases': 'Cassava Mosaic Disease, Cassava Brown Streak Disease'
            },
            {
                'name': 'Maize',
                'scientific_name': 'Zea mays',
                'local_names': '{"tw": "Aburo", "ga": "Aburo", "ha": "Masara", "ee": "Agbledela"}',
                'growing_season': 'March-July, September-December',
                'common_diseases': 'Maize Rust, Northern Corn Leaf Blight, Maize Smut'
            },
            {
                'name': 'Cocoa',
                'scientific_name': 'Theobroma cacao',
                'local_names': '{"tw": "Kokoo", "ga": "Kokoo", "ha": "Koko", "ee": "Kokoo"}',
                'growing_season': 'Year-round',
                'common_diseases': 'Cocoa Black Pod Disease, Cocoa Swollen Shoot Virus'
            },
            {
                'name': 'Yam',
                'scientific_name': 'Dioscorea spp.',
                'local_names': '{"tw": "Aberewa", "ga": "Aberewa", "ha": "Dankali", "ee": "Aberewa"}',
                'growing_season': 'March-October',
                'common_diseases': 'Yam Anthracnose, Yam Tuber Rot'
            },
            {
                'name': 'Plantain',
                'scientific_name': 'Musa spp.',
                'local_names': '{"tw": "Kokoo", "ga": "Kokoo", "ha": "Ayaba", "ee": "Kokoo"}',
                'growing_season': 'Year-round',
                'common_diseases': 'Black Sigatoka, Panama Disease'
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for crop in crops_data:
            cursor.execute('''
                INSERT OR IGNORE INTO crops (name, scientific_name, local_names, growing_season, common_diseases)
                VALUES (?, ?, ?, ?, ?)
            ''', (crop['name'], crop['scientific_name'], crop['local_names'], 
                  crop['growing_season'], crop['common_diseases']))
        
        conn.commit()
        conn.close()
    
    def _load_disease_data(self):
        """Load initial disease data."""
        diseases_data = [
            {
                'name': 'Cassava Mosaic Disease',
                'scientific_name': 'Cassava mosaic virus',
                'crop_type': 'cassava',
                'symptoms': 'Yellow mosaic patterns on leaves, stunted growth, reduced yield',
                'causes': 'Virus transmitted by whiteflies',
                'lifecycle': 'Virus infects plant through whitefly feeding, spreads systemically',
                'severity': 'high'
            },
            {
                'name': 'Maize Rust',
                'scientific_name': 'Puccinia sorghi',
                'crop_type': 'maize',
                'symptoms': 'Reddish-brown pustules on leaves, premature leaf death',
                'causes': 'Fungal pathogen, thrives in humid conditions',
                'lifecycle': 'Fungus overwinters on crop debris, spreads via wind-borne spores',
                'severity': 'moderate'
            },
            {
                'name': 'Cocoa Black Pod Disease',
                'scientific_name': 'Phytophthora palmivora',
                'crop_type': 'cocoa',
                'symptoms': 'Black lesions on pods, pod rot, yield loss',
                'causes': 'Fungal pathogen, spreads in wet conditions',
                'lifecycle': 'Fungus survives in soil, infects pods during wet weather',
                'severity': 'high'
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for disease in diseases_data:
            cursor.execute('''
                INSERT OR IGNORE INTO diseases (name, scientific_name, crop_type, symptoms, causes, lifecycle, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (disease['name'], disease['scientific_name'], disease['crop_type'],
                  disease['symptoms'], disease['causes'], disease['lifecycle'], disease['severity']))
        
        conn.commit()
        conn.close()
    
    def _load_local_practices(self):
        """Load local agricultural practices."""
        practices_data = [
            {
                'crop_name': 'Cassava',
                'practice_name': 'Crop Rotation',
                'description': 'Rotate cassava with legumes to improve soil fertility',
                'region': 'Ashanti, Eastern',
                'effectiveness': 0.8
            },
            {
                'crop_name': 'Maize',
                'practice_name': 'Early Planting',
                'description': 'Plant early in the season to avoid disease pressure',
                'region': 'Northern, Upper East',
                'effectiveness': 0.7
            },
            {
                'crop_name': 'Cocoa',
                'practice_name': 'Shade Management',
                'description': 'Maintain appropriate shade levels to reduce disease spread',
                'region': 'Western, Central',
                'effectiveness': 0.6
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for practice in practices_data:
            # Get crop ID
            cursor.execute('SELECT id FROM crops WHERE name = ?', (practice['crop_name'],))
            crop_result = cursor.fetchone()
            
            if crop_result:
                crop_id = crop_result[0]
                cursor.execute('''
                    INSERT OR IGNORE INTO local_practices (crop_id, practice_name, description, region, effectiveness)
                    VALUES (?, ?, ?, ?, ?)
                ''', (crop_id, practice['practice_name'], practice['description'],
                      practice['region'], practice['effectiveness']))
        
        conn.commit()
        conn.close()
    
    def get_disease_info(self, disease_name: str, crop_type: str = None) -> Dict[str, Any]:
        """
        Get comprehensive disease information.
        
        Args:
            disease_name: Name of the disease
            crop_type: Type of crop (optional)
            
        Returns:
            Disease information dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT d.*, GROUP_CONCAT(t.treatment_type) as treatments
            FROM diseases d
            LEFT JOIN treatments t ON d.id = t.disease_id
            WHERE d.name LIKE ? OR d.scientific_name LIKE ?
        '''
        
        if crop_type:
            query += ' AND d.crop_type = ?'
            cursor.execute(query, (f'%{disease_name}%', f'%{disease_name}%', crop_type))
        else:
            cursor.execute(query, (f'%{disease_name}%', f'%{disease_name}%'))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [description[0] for description in cursor.description]
            disease_info = dict(zip(columns, result))
            
            # Get treatments
            treatments = self.get_treatments_for_disease(disease_info['id'])
            disease_info['treatments'] = treatments
            
            return disease_info
        
        return {}
    
    def get_treatments_for_disease(self, disease_id: int) -> List[Dict[str, Any]]:
        """
        Get treatments for a specific disease.
        
        Args:
            disease_id: Disease ID
            
        Returns:
            List of treatment dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM treatments WHERE disease_id = ?
            ORDER BY effectiveness DESC
        ''', (disease_id,))
        
        treatments = []
        for row in cursor.fetchall():
            columns = [description[0] for description in cursor.description]
            treatment = dict(zip(columns, row))
            treatments.append(treatment)
        
        conn.close()
        return treatments
    
    def search_diseases(self, query: str, crop_type: str = None) -> List[Dict[str, Any]]:
        """
        Search for diseases based on query.
        
        Args:
            query: Search query
            crop_type: Filter by crop type
            
        Returns:
            List of matching diseases
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        search_query = '''
            SELECT * FROM diseases 
            WHERE (name LIKE ? OR symptoms LIKE ? OR causes LIKE ?)
        '''
        
        params = [f'%{query}%', f'%{query}%', f'%{query}%']
        
        if crop_type:
            search_query += ' AND crop_type = ?'
            params.append(crop_type)
        
        cursor.execute(search_query, params)
        
        diseases = []
        for row in cursor.fetchall():
            columns = [description[0] for description in cursor.description]
            disease = dict(zip(columns, row))
            diseases.append(disease)
        
        conn.close()
        return diseases
    
    def add_farmer_feedback(self, disease_id: int, feedback_text: str, 
                          feedback_type: str = 'correction', confidence_score: float = 0.5,
                          location: str = None, farmer_id: str = None) -> bool:
        """
        Add farmer feedback to the knowledge base.
        
        Args:
            disease_id: Disease ID
            feedback_text: Feedback text
            feedback_type: Type of feedback ('correction', 'confirmation', 'suggestion')
            confidence_score: Confidence score (0-1)
            location: Farmer location
            farmer_id: Farmer identifier
            
        Returns:
            True if feedback added successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO farmer_feedback 
                (disease_id, farmer_id, feedback_type, feedback_text, confidence_score, location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (disease_id, farmer_id, feedback_type, feedback_text, confidence_score, location))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Farmer feedback added for disease {disease_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add farmer feedback: {e}")
            return False
    
    def update_knowledge_base(self) -> Dict[str, Any]:
        """
        Update knowledge base with latest data.
        
        Returns:
            Update results summary
        """
        update_results = {
            'diseases_updated': 0,
            'treatments_updated': 0,
            'research_added': 0,
            'errors': []
        }
        
        try:
            # Update disease information
            diseases_updated = self.web_scraper.update_disease_data()
            update_results['diseases_updated'] = diseases_updated
            
            # Update treatment information
            treatments_updated = self.web_scraper.update_treatment_data()
            update_results['treatments_updated'] = treatments_updated
            
            # Add new research papers
            research_added = self.web_scraper.add_research_papers()
            update_results['research_added'] = research_added
            
            logger.info("Knowledge base updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            update_results['errors'].append(str(e))
        
        return update_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        tables = ['diseases', 'treatments', 'crops', 'local_practices', 'research_papers', 'farmer_feedback']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            stats[f'{table}_count'] = count
        
        # Get recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM farmer_feedback 
            WHERE created_date >= datetime('now', '-7 days')
        ''')
        recent_feedback = cursor.fetchone()[0]
        stats['recent_feedback'] = recent_feedback
        
        conn.close()
        return stats
    
    def export_data(self, format: str = 'json', file_path: str = None) -> bool:
        """
        Export knowledge base data.
        
        Args:
            format: Export format ('json', 'csv', 'sql')
            file_path: Output file path
            
        Returns:
            True if export successful
        """
        try:
            if format == 'json':
                return self._export_to_json(file_path)
            elif format == 'csv':
                return self._export_to_csv(file_path)
            elif format == 'sql':
                return self._export_to_sql(file_path)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False
    
    def _export_to_json(self, file_path: str) -> bool:
        """Export data to JSON format."""
        conn = sqlite3.connect(self.db_path)
        
        data = {}
        tables = ['diseases', 'treatments', 'crops', 'local_practices', 'research_papers']
        
        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
            data[table] = df.to_dict('records')
        
        conn.close()
        
        if file_path is None:
            file_path = f"exports/knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Data exported to {file_path}")
        return True
    
    def _export_to_csv(self, file_path: str) -> bool:
        """Export data to CSV format."""
        conn = sqlite3.connect(self.db_path)
        
        if file_path is None:
            file_path = f"exports/knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        tables = ['diseases', 'treatments', 'crops', 'local_practices', 'research_papers']
        
        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
            table_file = f"{file_path}_{table}.csv"
            df.to_csv(table_file, index=False)
        
        conn.close()
        logger.info(f"Data exported to {file_path}_*.csv files")
        return True
    
    def _export_to_sql(self, file_path: str) -> bool:
        """Export data to SQL format."""
        if file_path is None:
            file_path = f"exports/knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Use sqlite3 to export
        conn = sqlite3.connect(self.db_path)
        
        with open(file_path, 'w') as f:
            for line in conn.iterdump():
                f.write(f'{line}\n')
        
        conn.close()
        logger.info(f"Data exported to {file_path}")
        return True 
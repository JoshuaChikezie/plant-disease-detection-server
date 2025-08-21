#!/usr/bin/env python3
"""
Database Initialization Script for Plant Disease Detection System

This script initializes the database with:
- Sample disease data
- Treatment information
- Local agricultural practices
- Research papers
- User feedback examples
"""

import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ai_module.knowledge_base import KnowledgeBase


def init_database():
    """Initialize the database with sample data."""
    print("🌱 Initializing Plant Disease Detection Database...")
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize knowledge base
    db_path = data_dir / "knowledge_base.db"
    kb = KnowledgeBase(str(db_path))
    
    print("✅ Database initialized successfully!")
    
    # Add sample data
    add_sample_diseases(kb)
    add_sample_treatments(kb)
    add_sample_practices(kb)
    add_sample_research(kb)
    add_sample_feedback(kb)
    
    print("✅ Sample data added successfully!")
    
    # Display statistics
    stats = kb.get_statistics()
    print("\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def add_sample_diseases(kb):
    """Add sample disease data."""
    print("📝 Adding sample diseases...")
    
    diseases = [
        {
            'name': 'Cassava Mosaic Disease',
            'scientific_name': 'Cassava mosaic virus',
            'crop_type': 'cassava',
            'symptoms': 'Yellow mosaic patterns on leaves, stunted growth, reduced yield, leaf distortion',
            'causes': 'Virus transmitted by whiteflies (Bemisia tabaci), infected planting material',
            'lifecycle': 'Virus infects plant through whitefly feeding, spreads systemically through plant',
            'severity': 'high'
        },
        {
            'name': 'Cassava Brown Streak Disease',
            'scientific_name': 'Cassava brown streak virus',
            'crop_type': 'cassava',
            'symptoms': 'Brown streaks on stems, root necrosis, yield loss up to 100%',
            'causes': 'Virus transmitted by whiteflies, infected cuttings',
            'lifecycle': 'Virus spreads through infected planting material and whitefly vectors',
            'severity': 'very_high'
        },
        {
            'name': 'Maize Rust',
            'scientific_name': 'Puccinia sorghi',
            'crop_type': 'maize',
            'symptoms': 'Reddish-brown pustules on leaves, premature leaf death, reduced photosynthesis',
            'causes': 'Fungal pathogen, thrives in humid conditions, wind-borne spores',
            'lifecycle': 'Fungus overwinters on crop debris, spreads via wind-borne spores',
            'severity': 'moderate'
        },
        {
            'name': 'Northern Corn Leaf Blight',
            'scientific_name': 'Exserohilum turcicum',
            'crop_type': 'maize',
            'symptoms': 'Large, cigar-shaped lesions on leaves, gray to tan in color',
            'causes': 'Fungal pathogen, high humidity, moderate temperatures',
            'lifecycle': 'Fungus survives in crop residue, infects during wet weather',
            'severity': 'moderate'
        },
        {
            'name': 'Cocoa Black Pod Disease',
            'scientific_name': 'Phytophthora palmivora',
            'crop_type': 'cocoa',
            'symptoms': 'Black lesions on pods, pod rot, yield loss, tree dieback',
            'causes': 'Fungal pathogen, spreads in wet conditions, poor drainage',
            'lifecycle': 'Fungus survives in soil, infects pods during wet weather',
            'severity': 'high'
        },
        {
            'name': 'Cocoa Swollen Shoot Virus',
            'scientific_name': 'Cacao swollen shoot virus',
            'crop_type': 'cocoa',
            'symptoms': 'Swollen stems and roots, leaf chlorosis, reduced yield',
            'causes': 'Virus transmitted by mealybugs, infected planting material',
            'lifecycle': 'Virus spreads through mealybug vectors and infected trees',
            'severity': 'high'
        }
    ]
    
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    
    for disease in diseases:
        cursor.execute('''
            INSERT OR REPLACE INTO diseases 
            (name, scientific_name, crop_type, symptoms, causes, lifecycle, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (disease['name'], disease['scientific_name'], disease['crop_type'],
              disease['symptoms'], disease['causes'], disease['lifecycle'], disease['severity']))
    
    conn.commit()
    conn.close()


def add_sample_treatments(kb):
    """Add sample treatment data."""
    print("💊 Adding sample treatments...")
    
    treatments = [
        {
            'disease_name': 'Cassava Mosaic Disease',
            'treatment_type': 'Prevention',
            'description': 'Use virus-free planting material, control whitefly populations',
            'application_method': 'Plant certified disease-free cuttings, apply insecticides',
            'effectiveness': 0.8,
            'cost_category': 'low',
            'availability': 'high',
            'organic': True
        },
        {
            'disease_name': 'Cassava Mosaic Disease',
            'treatment_type': 'Chemical Control',
            'description': 'Apply systemic insecticides to control whitefly vectors',
            'application_method': 'Spray insecticides at recommended intervals',
            'effectiveness': 0.7,
            'cost_category': 'medium',
            'availability': 'medium',
            'organic': False
        },
        {
            'disease_name': 'Maize Rust',
            'treatment_type': 'Fungicide',
            'description': 'Apply fungicides at first sign of disease',
            'application_method': 'Spray fungicide on affected plants',
            'effectiveness': 0.6,
            'cost_category': 'medium',
            'availability': 'high',
            'organic': False
        },
        {
            'disease_name': 'Maize Rust',
            'treatment_type': 'Cultural Control',
            'description': 'Plant resistant varieties, practice crop rotation',
            'application_method': 'Select resistant maize varieties, rotate crops',
            'effectiveness': 0.7,
            'cost_category': 'low',
            'availability': 'high',
            'organic': True
        },
        {
            'disease_name': 'Cocoa Black Pod Disease',
            'treatment_type': 'Fungicide',
            'description': 'Apply copper-based fungicides',
            'application_method': 'Spray fungicide on pods and affected areas',
            'effectiveness': 0.8,
            'cost_category': 'medium',
            'availability': 'high',
            'organic': False
        },
        {
            'disease_name': 'Cocoa Black Pod Disease',
            'treatment_type': 'Cultural Control',
            'description': 'Improve drainage, remove infected pods',
            'application_method': 'Ensure good drainage, prune affected areas',
            'effectiveness': 0.6,
            'cost_category': 'low',
            'availability': 'high',
            'organic': True
        }
    ]
    
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    
    for treatment in treatments:
        # Get disease ID
        cursor.execute('SELECT id FROM diseases WHERE name = ?', (treatment['disease_name'],))
        disease_result = cursor.fetchone()
        
        if disease_result:
            disease_id = disease_result[0]
            cursor.execute('''
                INSERT OR REPLACE INTO treatments 
                (disease_id, treatment_type, description, application_method, 
                 effectiveness, cost_category, availability, organic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (disease_id, treatment['treatment_type'], treatment['description'],
                  treatment['application_method'], treatment['effectiveness'],
                  treatment['cost_category'], treatment['availability'], treatment['organic']))
    
    conn.commit()
    conn.close()


def add_sample_practices(kb):
    """Add sample local agricultural practices."""
    print("🌾 Adding sample local practices...")
    
    practices = [
        {
            'crop_name': 'Cassava',
            'practice_name': 'Crop Rotation',
            'description': 'Rotate cassava with legumes to improve soil fertility and break disease cycles',
            'region': 'Ashanti, Eastern, Central',
            'effectiveness': 0.8
        },
        {
            'crop_name': 'Cassava',
            'practice_name': 'Early Planting',
            'description': 'Plant early in the season to avoid peak disease pressure',
            'region': 'All regions',
            'effectiveness': 0.7
        },
        {
            'crop_name': 'Maize',
            'practice_name': 'Intercropping',
            'description': 'Intercrop maize with beans or cowpea to reduce disease spread',
            'region': 'Northern, Upper East, Upper West',
            'effectiveness': 0.6
        },
        {
            'crop_name': 'Maize',
            'practice_name': 'Resistant Varieties',
            'description': 'Use disease-resistant maize varieties',
            'region': 'All regions',
            'effectiveness': 0.9
        },
        {
            'crop_name': 'Cocoa',
            'practice_name': 'Shade Management',
            'description': 'Maintain appropriate shade levels to reduce disease spread',
            'region': 'Western, Central, Ashanti',
            'effectiveness': 0.6
        },
        {
            'crop_name': 'Cocoa',
            'practice_name': 'Sanitation',
            'description': 'Remove and destroy infected pods and plant debris',
            'region': 'All cocoa growing regions',
            'effectiveness': 0.7
        }
    ]
    
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    
    for practice in practices:
        # Get crop ID
        cursor.execute('SELECT id FROM crops WHERE name = ?', (practice['crop_name'],))
        crop_result = cursor.fetchone()
        
        if crop_result:
            crop_id = crop_result[0]
            cursor.execute('''
                INSERT OR REPLACE INTO local_practices 
                (crop_id, practice_name, description, region, effectiveness)
                VALUES (?, ?, ?, ?, ?)
            ''', (crop_id, practice['practice_name'], practice['description'],
                  practice['region'], practice['effectiveness']))
    
    conn.commit()
    conn.close()


def add_sample_research(kb):
    """Add sample research papers."""
    print("📚 Adding sample research papers...")
    
    research_papers = [
        {
            'title': 'Cassava Mosaic Disease Management in Ghana',
            'authors': 'Kwame Asante, Abena Osei, Kofi Mensah',
            'abstract': 'Study on effective management strategies for cassava mosaic disease in Ghanaian farms',
            'publication_date': '2023-06-15',
            'source': 'Ghana Journal of Agricultural Science',
            'url': 'https://example.com/paper1',
            'keywords': 'cassava, mosaic disease, Ghana, management'
        },
        {
            'title': 'Maize Rust Resistance in Local Varieties',
            'authors': 'Sarah Addo, Emmanuel Kwarteng',
            'abstract': 'Evaluation of local maize varieties for rust resistance',
            'publication_date': '2023-08-20',
            'source': 'African Crop Science Journal',
            'url': 'https://example.com/paper2',
            'keywords': 'maize, rust, resistance, local varieties'
        },
        {
            'title': 'Cocoa Black Pod Disease Control Methods',
            'authors': 'Daniel Owusu, Grace Ampah',
            'abstract': 'Comparison of chemical and biological control methods for cocoa black pod disease',
            'publication_date': '2023-09-10',
            'source': 'International Journal of Plant Pathology',
            'url': 'https://example.com/paper3',
            'keywords': 'cocoa, black pod, control, fungicides'
        }
    ]
    
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    
    for paper in research_papers:
        cursor.execute('''
            INSERT OR REPLACE INTO research_papers 
            (title, authors, abstract, publication_date, source, url, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (paper['title'], paper['authors'], paper['abstract'],
              paper['publication_date'], paper['source'], paper['url'], paper['keywords']))
    
    conn.commit()
    conn.close()


def add_sample_feedback(kb):
    """Add sample farmer feedback."""
    print("👨‍🌾 Adding sample farmer feedback...")
    
    feedback_data = [
        {
            'disease_name': 'Cassava Mosaic Disease',
            'farmer_id': 'FARMER001',
            'feedback_type': 'confirmation',
            'feedback_text': 'The app correctly identified cassava mosaic disease in my farm',
            'confidence_score': 0.9,
            'location': 'Kumasi, Ashanti Region'
        },
        {
            'disease_name': 'Maize Rust',
            'farmer_id': 'FARMER002',
            'feedback_type': 'correction',
            'feedback_text': 'The app suggested maize rust but it was actually northern leaf blight',
            'confidence_score': 0.7,
            'location': 'Tamale, Northern Region'
        },
        {
            'disease_name': 'Cocoa Black Pod Disease',
            'farmer_id': 'FARMER003',
            'feedback_type': 'suggestion',
            'feedback_text': 'Add more information about organic treatment options',
            'confidence_score': 0.8,
            'location': 'Tarkwa, Western Region'
        }
    ]
    
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    
    for feedback in feedback_data:
        # Get disease ID
        cursor.execute('SELECT id FROM diseases WHERE name = ?', (feedback['disease_name'],))
        disease_result = cursor.fetchone()
        
        if disease_result:
            disease_id = disease_result[0]
            cursor.execute('''
                INSERT OR REPLACE INTO farmer_feedback 
                (disease_id, farmer_id, feedback_type, feedback_text, confidence_score, location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (disease_id, feedback['farmer_id'], feedback['feedback_type'],
                  feedback['feedback_text'], feedback['confidence_score'], feedback['location']))
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_database()
    print("\n🎉 Database initialization completed successfully!")
    print("You can now start the API server with: python api/main.py") 
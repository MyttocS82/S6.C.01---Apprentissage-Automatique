"""
Configuration settings for Yelp dataset analysis
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Yelp dataset files (JSON format from Yelp Open Dataset)
YELP_BUSINESS_FILE = RAW_DATA_DIR / "yelp_academic_dataset_business.json"
YELP_REVIEW_FILE = RAW_DATA_DIR / "yelp_academic_dataset_review.json"
YELP_USER_FILE = RAW_DATA_DIR / "yelp_academic_dataset_user.json"

# Rating scale
MIN_STARS = 1
MAX_STARS = 5

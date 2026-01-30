"""
Data loader for Yelp Open Dataset
Handles loading and parsing of Yelp JSON datasets
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
from tqdm import tqdm


def load_json_lines(file_path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Load a JSON lines file (one JSON object per line)
    
    Args:
        file_path: Path to the JSON lines file
        limit: Maximum number of lines to load (None for all)
        
    Yields:
        Dictionary representing each JSON object
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def load_business_data(file_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load Yelp business data into a pandas DataFrame
    
    Args:
        file_path: Path to the business JSON file
        limit: Maximum number of businesses to load
        
    Returns:
        DataFrame containing business information
    """
    businesses = list(load_json_lines(file_path, limit))
    df = pd.DataFrame(businesses)
    
    # Convert stars to numeric if present
    if 'stars' in df.columns:
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    
    return df


def load_review_data(file_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load Yelp review data into a pandas DataFrame
    
    Args:
        file_path: Path to the review JSON file
        limit: Maximum number of reviews to load
        
    Returns:
        DataFrame containing review information
    """
    print(f"Loading reviews from {file_path}...")
    reviews = []
    
    for review in tqdm(load_json_lines(file_path, limit), desc="Loading reviews"):
        reviews.append(review)
    
    df = pd.DataFrame(reviews)
    
    # Convert stars to numeric if present
    if 'stars' in df.columns:
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    
    # Convert date to datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df


def load_user_data(file_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load Yelp user data into a pandas DataFrame
    
    Args:
        file_path: Path to the user JSON file
        limit: Maximum number of users to load
        
    Returns:
        DataFrame containing user information
    """
    users = list(load_json_lines(file_path, limit))
    df = pd.DataFrame(users)
    
    # Convert date to datetime if present
    if 'yelping_since' in df.columns:
        df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')
    
    return df


class YelpDataLoader:
    """
    Unified loader for Yelp Open Dataset
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the Yelp dataset files
        """
        self.data_dir = data_dir
        self.business_file = data_dir / "yelp_academic_dataset_business.json"
        self.review_file = data_dir / "yelp_academic_dataset_review.json"
        self.user_file = data_dir / "yelp_academic_dataset_user.json"
    
    def load_businesses(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load business data"""
        return load_business_data(self.business_file, limit)
    
    def load_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load review data"""
        return load_review_data(self.review_file, limit)
    
    def load_users(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load user data"""
        return load_user_data(self.user_file, limit)

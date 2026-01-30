"""
Review analysis utilities for Yelp dataset
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class ReviewAnalyzer:
    """
    Analyze reviews from Yelp dataset
    """
    
    def __init__(self, reviews_df: pd.DataFrame):
        """
        Initialize the review analyzer
        
        Args:
            reviews_df: DataFrame containing review data
        """
        self.reviews = reviews_df
    
    def get_rating_distribution(self) -> pd.Series:
        """
        Get distribution of star ratings
        
        Returns:
            Series with count of reviews for each star rating (1-5)
        """
        if 'stars' not in self.reviews.columns:
            return pd.Series()
        
        return self.reviews['stars'].value_counts().sort_index()
    
    def get_average_rating(self) -> float:
        """
        Calculate average rating across all reviews
        
        Returns:
            Average star rating
        """
        if 'stars' not in self.reviews.columns:
            return 0.0
        
        return self.reviews['stars'].mean()
    
    def filter_by_rating(self, stars: int) -> pd.DataFrame:
        """
        Filter reviews by specific star rating
        
        Args:
            stars: Star rating to filter by (1-5)
            
        Returns:
            DataFrame of reviews with specified rating
        """
        if 'stars' not in self.reviews.columns:
            return pd.DataFrame()
        
        return self.reviews[self.reviews['stars'] == stars]
    
    def get_reviews_by_business(self, business_id: str) -> pd.DataFrame:
        """
        Get all reviews for a specific business
        
        Args:
            business_id: Business ID to filter by
            
        Returns:
            DataFrame of reviews for the business
        """
        if 'business_id' not in self.reviews.columns:
            return pd.DataFrame()
        
        return self.reviews[self.reviews['business_id'] == business_id]
    
    def get_reviews_by_user(self, user_id: str) -> pd.DataFrame:
        """
        Get all reviews by a specific user
        
        Args:
            user_id: User ID to filter by
            
        Returns:
            DataFrame of reviews by the user
        """
        if 'user_id' not in self.reviews.columns:
            return pd.DataFrame()
        
        return self.reviews[self.reviews['user_id'] == user_id]
    
    def get_text_length_stats(self) -> Dict[str, float]:
        """
        Get statistics about review text lengths
        
        Returns:
            Dictionary with text length statistics
        """
        if 'text' not in self.reviews.columns:
            return {}
        
        lengths = self.reviews['text'].str.len()
        
        return {
            'mean': lengths.mean(),
            'median': lengths.median(),
            'min': lengths.min(),
            'max': lengths.max(),
            'std': lengths.std()
        }
    
    def get_most_useful_reviews(self, n: int = 10) -> pd.DataFrame:
        """
        Get most useful reviews (by useful votes)
        
        Args:
            n: Number of reviews to return
            
        Returns:
            DataFrame of most useful reviews
        """
        if 'useful' not in self.reviews.columns:
            return pd.DataFrame()
        
        return self.reviews.nlargest(n, 'useful')
    
    def analyze_sentiment_by_rating(self) -> pd.DataFrame:
        """
        Analyze relationship between ratings and review characteristics
        
        Returns:
            DataFrame with statistics grouped by rating
        """
        if 'stars' not in self.reviews.columns:
            return pd.DataFrame()
        
        stats = []
        for rating in range(1, 6):
            rating_reviews = self.filter_by_rating(rating)
            if len(rating_reviews) > 0:
                stats.append({
                    'stars': rating,
                    'count': len(rating_reviews),
                    'avg_text_length': rating_reviews['text'].str.len().mean() if 'text' in rating_reviews.columns else 0,
                    'avg_useful': rating_reviews['useful'].mean() if 'useful' in rating_reviews.columns else 0,
                    'avg_funny': rating_reviews['funny'].mean() if 'funny' in rating_reviews.columns else 0,
                    'avg_cool': rating_reviews['cool'].mean() if 'cool' in rating_reviews.columns else 0,
                })
        
        return pd.DataFrame(stats)

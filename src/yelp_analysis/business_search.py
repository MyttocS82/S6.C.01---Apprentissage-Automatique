"""
Business search and filtering functionality for Yelp dataset
"""
import pandas as pd
from typing import List, Optional


class BusinessSearch:
    """
    Search and filter businesses from Yelp dataset
    """
    
    def __init__(self, businesses_df: pd.DataFrame):
        """
        Initialize the business search
        
        Args:
            businesses_df: DataFrame containing business data
        """
        self.businesses = businesses_df
    
    def search_by_name(self, query: str, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search businesses by name
        
        Args:
            query: Search query string
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame of matching businesses
        """
        if not case_sensitive:
            mask = self.businesses['name'].str.contains(query, case=False, na=False)
        else:
            mask = self.businesses['name'].str.contains(query, na=False)
        
        return self.businesses[mask]
    
    def search_by_category(self, category: str) -> pd.DataFrame:
        """
        Search businesses by category
        
        Args:
            category: Category to search for (e.g., 'Restaurants', 'Hotels')
            
        Returns:
            DataFrame of matching businesses
        """
        if 'categories' not in self.businesses.columns:
            return pd.DataFrame()
        
        mask = self.businesses['categories'].str.contains(category, case=False, na=False)
        return self.businesses[mask]
    
    def filter_by_rating(self, min_stars: float = 1.0, max_stars: float = 5.0) -> pd.DataFrame:
        """
        Filter businesses by rating range
        
        Args:
            min_stars: Minimum star rating (1-5)
            max_stars: Maximum star rating (1-5)
            
        Returns:
            DataFrame of businesses within rating range
        """
        if 'stars' not in self.businesses.columns:
            return pd.DataFrame()
        
        mask = (self.businesses['stars'] >= min_stars) & (self.businesses['stars'] <= max_stars)
        return self.businesses[mask]
    
    def filter_by_location(self, city: Optional[str] = None, state: Optional[str] = None) -> pd.DataFrame:
        """
        Filter businesses by location
        
        Args:
            city: City name to filter by
            state: State code to filter by
            
        Returns:
            DataFrame of businesses in the specified location
        """
        result = self.businesses
        
        if city and 'city' in result.columns:
            result = result[result['city'].str.contains(city, case=False, na=False)]
        
        if state and 'state' in result.columns:
            result = result[result['state'] == state]
        
        return result
    
    def get_top_rated(self, n: int = 10, category: Optional[str] = None) -> pd.DataFrame:
        """
        Get top-rated businesses
        
        Args:
            n: Number of top businesses to return
            category: Optional category filter
            
        Returns:
            DataFrame of top-rated businesses
        """
        result = self.businesses
        
        if category:
            result = self.search_by_category(category)
        
        if 'stars' in result.columns:
            result = result.sort_values('stars', ascending=False)
        
        return result.head(n)

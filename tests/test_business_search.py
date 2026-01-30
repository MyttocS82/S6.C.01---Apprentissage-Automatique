"""
Tests for business_search module
"""
import unittest
import pandas as pd
from yelp_analysis.business_search import BusinessSearch


class TestBusinessSearch(unittest.TestCase):
    """Test cases for BusinessSearch class"""
    
    def setUp(self):
        """Set up test data"""
        self.test_businesses = pd.DataFrame([
            {
                'business_id': 'b1',
                'name': 'Restaurant Italien',
                'stars': 4.5,
                'categories': 'Restaurants, Italian, Pizza',
                'city': 'Phoenix',
                'state': 'AZ'
            },
            {
                'business_id': 'b2',
                'name': 'Hotel Paradise',
                'stars': 5.0,
                'categories': 'Hotels, Travel',
                'city': 'Phoenix',
                'state': 'AZ'
            },
            {
                'business_id': 'b3',
                'name': 'Bar Central',
                'stars': 3.5,
                'categories': 'Bars, Nightlife',
                'city': 'Las Vegas',
                'state': 'NV'
            }
        ])
        self.search = BusinessSearch(self.test_businesses)
    
    def test_search_by_name(self):
        """Test search by business name"""
        result = self.search.search_by_name('Hotel')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['name'], 'Hotel Paradise')
    
    def test_search_by_category(self):
        """Test search by category"""
        result = self.search.search_by_category('Restaurants')
        self.assertEqual(len(result), 1)
        self.assertIn('Restaurant Italien', result['name'].values)
    
    def test_filter_by_rating(self):
        """Test filtering by rating"""
        result = self.search.filter_by_rating(min_stars=4.0, max_stars=5.0)
        self.assertEqual(len(result), 2)
    
    def test_filter_by_location(self):
        """Test filtering by location"""
        result = self.search.filter_by_location(city='Phoenix')
        self.assertEqual(len(result), 2)
        
        result = self.search.filter_by_location(state='NV')
        self.assertEqual(len(result), 1)
    
    def test_get_top_rated(self):
        """Test getting top-rated businesses"""
        result = self.search.get_top_rated(n=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['stars'], 5.0)


if __name__ == '__main__':
    unittest.main()

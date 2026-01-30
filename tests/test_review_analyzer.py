"""
Tests for review_analyzer module
"""
import unittest
import pandas as pd
from yelp_analysis.review_analyzer import ReviewAnalyzer


class TestReviewAnalyzer(unittest.TestCase):
    """Test cases for ReviewAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.test_reviews = pd.DataFrame([
            {
                'review_id': 'r1',
                'business_id': 'b1',
                'user_id': 'u1',
                'stars': 5,
                'text': 'Excellent restaurant!',
                'useful': 10,
                'funny': 2,
                'cool': 5
            },
            {
                'review_id': 'r2',
                'business_id': 'b1',
                'user_id': 'u2',
                'stars': 4,
                'text': 'Good food, nice service.',
                'useful': 5,
                'funny': 0,
                'cool': 2
            },
            {
                'review_id': 'r3',
                'business_id': 'b2',
                'user_id': 'u1',
                'stars': 3,
                'text': 'Average experience.',
                'useful': 2,
                'funny': 1,
                'cool': 1
            },
            {
                'review_id': 'r4',
                'business_id': 'b2',
                'user_id': 'u3',
                'stars': 5,
                'text': 'Amazing!',
                'useful': 15,
                'funny': 3,
                'cool': 8
            }
        ])
        self.analyzer = ReviewAnalyzer(self.test_reviews)
    
    def test_get_rating_distribution(self):
        """Test getting rating distribution"""
        distribution = self.analyzer.get_rating_distribution()
        self.assertEqual(distribution[5], 2)
        self.assertEqual(distribution[4], 1)
        self.assertEqual(distribution[3], 1)
    
    def test_get_average_rating(self):
        """Test calculating average rating"""
        avg = self.analyzer.get_average_rating()
        self.assertAlmostEqual(avg, 4.25, places=2)
    
    def test_filter_by_rating(self):
        """Test filtering by specific rating"""
        five_star_reviews = self.analyzer.filter_by_rating(5)
        self.assertEqual(len(five_star_reviews), 2)
    
    def test_get_reviews_by_business(self):
        """Test getting reviews for a specific business"""
        business_reviews = self.analyzer.get_reviews_by_business('b1')
        self.assertEqual(len(business_reviews), 2)
    
    def test_get_reviews_by_user(self):
        """Test getting reviews by a specific user"""
        user_reviews = self.analyzer.get_reviews_by_user('u1')
        self.assertEqual(len(user_reviews), 2)
    
    def test_get_most_useful_reviews(self):
        """Test getting most useful reviews"""
        most_useful = self.analyzer.get_most_useful_reviews(n=2)
        self.assertEqual(len(most_useful), 2)
        self.assertEqual(most_useful.iloc[0]['useful'], 15)
    
    def test_get_text_length_stats(self):
        """Test getting text length statistics"""
        stats = self.analyzer.get_text_length_stats()
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertGreater(stats['mean'], 0)


if __name__ == '__main__':
    unittest.main()

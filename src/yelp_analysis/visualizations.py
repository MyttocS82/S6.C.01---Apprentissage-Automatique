"""
Visualization utilities for Yelp data analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_rating_distribution(ratings: pd.Series, title: str = "Distribution des étoiles (1-5)"):
    """
    Plot distribution of star ratings
    
    Args:
        ratings: Series containing star ratings
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    ratings.value_counts().sort_index().plot(kind='bar', color='gold', edgecolor='black')
    plt.xlabel("Nombre d'étoiles")
    plt.ylabel("Nombre d'avis")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_business_categories(businesses_df: pd.DataFrame, top_n: int = 10):
    """
    Plot top business categories
    
    Args:
        businesses_df: DataFrame with business data
        top_n: Number of top categories to show
    """
    if 'categories' not in businesses_df.columns:
        print("No categories column found")
        return
    
    # Extract and count categories
    all_categories = []
    for cats in businesses_df['categories'].dropna():
        if isinstance(cats, str):
            all_categories.extend([c.strip() for c in cats.split(',')])
    
    category_counts = pd.Series(all_categories).value_counts().head(top_n)
    
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='barh', color='steelblue')
    plt.xlabel("Nombre d'établissements")
    plt.ylabel("Catégorie")
    plt.title(f"Top {top_n} des catégories d'établissements")
    plt.tight_layout()
    return plt.gcf()


def plot_rating_vs_review_count(businesses_df: pd.DataFrame):
    """
    Plot relationship between average rating and number of reviews
    
    Args:
        businesses_df: DataFrame with business data
    """
    if 'stars' not in businesses_df.columns or 'review_count' not in businesses_df.columns:
        print("Required columns not found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(businesses_df['review_count'], businesses_df['stars'], alpha=0.5)
    plt.xlabel("Nombre d'avis")
    plt.ylabel("Note moyenne (étoiles)")
    plt.title("Relation entre le nombre d'avis et la note moyenne")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_reviews_over_time(reviews_df: pd.DataFrame):
    """
    Plot number of reviews over time
    
    Args:
        reviews_df: DataFrame with review data
    """
    if 'date' not in reviews_df.columns:
        print("No date column found")
        return
    
    # Ensure date column is datetime and set as index
    reviews_df_copy = reviews_df.copy()
    reviews_df_copy['date'] = pd.to_datetime(reviews_df_copy['date'], errors='coerce')
    reviews_by_date = reviews_df_copy.set_index('date').resample('M').size()
    
    plt.figure(figsize=(14, 6))
    reviews_by_date.plot(color='darkgreen')
    plt.xlabel("Date")
    plt.ylabel("Nombre d'avis")
    plt.title("Évolution du nombre d'avis au fil du temps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

from src.data.load_data import load_yelp_datasets, load_processed_datasets, write_dataframe_to_csv
from src.utils.utils import create_sentiment_and_rating_columns, count_number_elements
from pathlib import Path
import pandas as pd

RAW_DATA_FOLDER = Path("../../data/raw")
PROCESSED_DATA_FOLDER = Path("../../data/processed")
ML_DATA_FOLDER = Path("../../data/ml")


def preprocess_datasets():
    """
    Preprocess raw Yelp datasets and save the processed datasets to CSV files.
    :return: None
    """
    # Loading raw datasets
    business, users, reviews, test = load_yelp_datasets(RAW_DATA_FOLDER)

    """
    Preprocessing des datasets brutes en datasets légèrement pré-traités
    """
    ''' Preprocessing Business DataFrame '''
    # Drop les colonnes inutiles        (garder 'categories' ?)
    business = business.drop(columns=['address', 'state', 'postal_code', 'latitude',
                                      'longitude', 'attributes', 'hours'])

    ''' Preprocessing Users DataFrame '''
    # Ajout d'une colonne du nombre d'années 'elite' et du nombre d'amis
    users['elite_years_count'] = users['elite'].apply(count_number_elements)
    users['friends_count'] = users['friends'].apply(count_number_elements)

    # Ajout d'une colonne du nombre total de compliments reçus
    compliment_columns = ['compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                          'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                          'compliment_funny', 'compliment_writer', 'compliment_photos']
    users['total_compliments'] = users[compliment_columns].sum(axis=1)

    # Ajout d'une colonne 'ancienneté' (en années)
    current_year = pd.Timestamp.now().year
    users['account_age_years'] = current_year - pd.to_datetime(users['yelping_since']).dt.year

    # Drop les colonnes inutiles
    users = users.drop(columns=['name', 'elite', 'friends', 'yelping_since'] + compliment_columns)

    ''' Preprocessing Reviews DataFrame  '''
    # Drop les colonnes inutiles
    reviews = reviews.drop(columns=['date'])
    test = test.drop(columns=['date'])

    # Création des labels pour le sentiment : score > 3 → positif // score < 3 → négatif // score = 3 → neutre
    #   et création d'une colonne 'rating' (eq à stars)
    create_sentiment_and_rating_columns(reviews)
    create_sentiment_and_rating_columns(test)

    ''' Écriture des DataFrames préprocessés dans des fichiers CSV '''
    write_dataframe_to_csv(business, PROCESSED_DATA_FOLDER, 'processed_business.csv')
    write_dataframe_to_csv(users, PROCESSED_DATA_FOLDER, 'processed_users.csv')
    write_dataframe_to_csv(reviews, PROCESSED_DATA_FOLDER, 'processed_reviews.csv')
    write_dataframe_to_csv(test, PROCESSED_DATA_FOLDER, 'test_dataset.csv')


def ml_datasets():
    """
    Create datasets ready for machine learning tasks and save them to CSV files.
    :return: None
    """
    # Loading processed datasets
    processed_reviews = load_processed_datasets(PROCESSED_DATA_FOLDER, "processed_reviews.csv")
    processed_test = load_processed_datasets(PROCESSED_DATA_FOLDER, 'test_dataset.csv')

    ''' Dataset reviews pour le Machine Learning '''
    processed_reviews.drop(columns=['business_id', 'review_id', 'user_id', 'stars', 'useful',
                                    'funny', 'cool'], inplace=True)

    processed_test.drop(columns=['business_id', 'review_id', 'user_id', 'stars', 'useful', 'funny', 'cool'],
                        inplace=True)

    dataset_sentiment = processed_reviews.copy().drop(columns=['rating'])
    dataset_rating = processed_reviews.copy().drop(columns=['sentiment'])

    test_sentiment = processed_test.copy().drop(columns=['rating'])
    test_rating = processed_test.copy().drop(columns=['sentiment'])

    ''' Écriture des DataFrames pour le ML dans des fichiers CSV '''
    write_dataframe_to_csv(dataset_sentiment, ML_DATA_FOLDER, 'ml_reviews_sentiment.csv')
    write_dataframe_to_csv(dataset_rating, ML_DATA_FOLDER, 'ml_reviews_rating.csv')

    write_dataframe_to_csv(test_sentiment, ML_DATA_FOLDER, 'ml_test_sentiment.csv')
    write_dataframe_to_csv(test_rating, ML_DATA_FOLDER, 'ml_test_rating.csv')


if __name__ == "__main__":
    preprocess_datasets()
    ml_datasets()

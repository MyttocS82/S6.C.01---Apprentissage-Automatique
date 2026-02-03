from src.data.load_data import load_yelp_datasets, write_dataframe_to_csv
from src.utils.utils import create_label_review, create_sentiment_and_rating_columns
from pandas import DataFrame
from pathlib import Path
import pandas as pd

raw_data_folder = Path("../../data/raw")
processed_data_folder = Path("../../data/processed")

buisness, users, reviews = load_yelp_datasets(raw_data_folder)

'''
Preprocessing Business DataFrame
'''
# Drop les colonnes inutiles        (garder 'categories' ?)
buisness = buisness.drop(columns=['business_id', 'address', 'state', 'postal_code', 'latitude',
                                  'longitude', 'attributes', 'hours'])
create_sentiment_and_rating_columns(buisness)   # Utile ?

print(buisness.info())

'''
Preprocessing Users DataFrame TODO
'''
# TODO

'''
Preprocessing Reviews DataFrame
'''
# Drop les colonnes inutiles
reviews = reviews.drop(columns=['user_id', 'business_id', 'review_id', 'date'])

# Création des labels pour le sentiment : score > 3 → positif // score < 3 → négatif // score = 3 → neutre
#   et création d'une colonne 'rating' (eq à stars)
create_sentiment_and_rating_columns(reviews)

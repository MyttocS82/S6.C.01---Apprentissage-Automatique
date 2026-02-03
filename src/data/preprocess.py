from src.data.load_data import load_yelp_datasets, write_dataframe_to_csv
from pandas import DataFrame
from pathlib import Path
import pandas as pd

raw_data_folder = Path("../../data/raw")
processed_data_folder = Path("../../data/processed")

buisness, users, reviews = load_yelp_datasets(raw_data_folder)

def create_label_review(stars: int) -> int:
    """
    Label the review based on the star rating.
    :param stars:
    :return: 1 if stars > 3 (positif), -1 if stars < 3 (négatif), 0 if stars == 3 (neutre)
    """
    if stars > 3:
        return 1
    elif stars < 3:
        return -1
    else:
        return 0

'''
Preprocessing Business DataFrame
'''
# Drop les colonnes inutiles        (garder 'categories' ?)
buisness = buisness.drop(columns=['business_id', 'address', 'state', 'postal_code', 'latitude',
                                  'longitude', 'attributes', 'hours'])

print(buisness.info())


'''
Preprocessing Users DataFrame TODO
'''


'''
Preprocessing Reviews DataFrame
'''
# Drop les colonnes inutiles
reviews = reviews.drop(columns=['user_id', 'business_id', 'review_id', 'date'])

# Création des labels pour le sentiment : score > 3 → positif // score < 3 → négatif // score = 3 → neutre
#   et création d'une colonne 'rating' (eq à stars)
reviews['sentiment'] = reviews['stars'].apply(create_label_review)
reviews['rating'] = reviews['stars']

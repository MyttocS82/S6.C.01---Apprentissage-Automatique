from src.data.load_data import load_yelp_datasets, write_dataframe_to_csv
from src.utils.utils import create_label_review, create_sentiment_and_rating_columns, count_number_elements
from pandas import DataFrame
from pathlib import Path
import pandas as pd

raw_data_folder = Path("../../data/raw")
processed_data_folder = Path("../../data/processed")

business, users, reviews = load_yelp_datasets(raw_data_folder)

'''
Preprocessing Business DataFrame
'''
# Drop les colonnes inutiles        (garder 'categories' ?)
business = business.drop(columns=['business_id', 'address', 'state', 'postal_code', 'latitude',
                                  'longitude', 'attributes', 'hours'])
create_sentiment_and_rating_columns(business)  # Utile ?

'''
Preprocessing Users DataFrame TODO
'''
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
users = users.drop(columns=['user_id', 'name', 'elite', 'friends', 'yelping_since'] + compliment_columns)

'''
Preprocessing Reviews DataFrame
'''
# Drop les colonnes inutiles
reviews = reviews.drop(columns=['user_id', 'business_id', 'review_id', 'date'])

# Création des labels pour le sentiment : score > 3 → positif // score < 3 → négatif // score = 3 → neutre
#   et création d'une colonne 'rating' (eq à stars)
create_sentiment_and_rating_columns(reviews)

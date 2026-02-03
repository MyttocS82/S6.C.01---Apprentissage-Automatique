from pandas import DataFrame


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


def create_sentiment_and_rating_columns(df: DataFrame) -> None:
    """
    Create 'sentiment' and 'rating' columns in the DataFrame based on the 'stars' column.
    :param df: DataFrame with a 'stars' column
    :return: DataFrame with added 'sentiment' and 'rating' columns
    """
    df['sentiment'] = df['stars'].apply(create_label_review)
    df['rating'] = df['stars']

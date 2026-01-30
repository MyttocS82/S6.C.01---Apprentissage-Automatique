from pathlib import Path
import pandas as pd
from pandas import DataFrame


def load_yelp_datasets(path_directory: Path) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load Yelp datasets from the specified directory.
    :param path_directory:
    :return: A tuple containing three DataFrames:
        1. business_dataset
        2. user_dataset
        3. reviews_dataset
    Raises FileNotFoundError if any of the required files are missing.
    """

    business_dataset = None
    user_dataset = None
    reviews_dataset = None

    for file in path_directory.iterdir():
        if file.name == "yelp_academic_dataset_business.json":
            business_dataset = pd.read_json(file, lines=True)

        elif file.name == "yelp_academic_dataset_user4students.jsonl":
            user_dataset = pd.read_json(file, lines=True)

        elif file.name == "yelp_academic_reviews4students.jsonl":
            reviews_dataset = pd.read_json(file, lines=True)

    if business_dataset is None or user_dataset is None or reviews_dataset is None:
        raise FileNotFoundError("One or more required Yelp dataset files are missing.")

    return business_dataset, user_dataset, reviews_dataset

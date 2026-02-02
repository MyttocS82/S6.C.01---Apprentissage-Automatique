from pandas import DataFrame
from pathlib import Path
import pandas as pd


def load_yelp_datasets(path_directory: Path) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load raw Yelp datasets from the specified directory.
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


def write_dataframe_to_csv(dataframe: DataFrame, output_path: Path) -> None:
    """
    Write a DataFrame to a CSV file at the specified output path.
    :param dataframe: The DataFrame to write.
    :param output_path: The path where the CSV file will be saved.
    """
    try:
        if not output_path.exists():
            raise FileNotFoundError("The folder does not exist.")
        dataframe.to_csv(output_path, index=False)
    except Exception as e:
        print(f"An error occurred while writing the DataFrame to CSV: {e}")


def load_processed_datasets(path_directory: Path) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load processed Yelp datasets from the specified directory.
    :param path_directory:
    :return: A tuple containing three DataFrames:
        1. processed_business_dataset
        2. processed_user_dataset
        3. processed_reviews_dataset
    Raises FileNotFoundError if any of the required files are missing.
    """
    processed_business_dataset = None
    processed_user_dataset = None
    processed_reviews_dataset = None

    for file in path_directory.iterdir():
        if file.name == "processed_business_dataset.csv":
            processed_business_dataset = pd.read_csv(file)

        elif file.name == "processed_user_dataset.csv":
            processed_user_dataset = pd.read_csv(file)

        elif file.name == "processed_reviews_dataset.csv":
            processed_reviews_dataset = pd.read_csv(file)

    if processed_business_dataset is None or processed_user_dataset is None or processed_reviews_dataset is None:
        raise FileNotFoundError("One or more required processed Yelp dataset files are missing.")

    return processed_business_dataset, processed_user_dataset, processed_reviews_dataset

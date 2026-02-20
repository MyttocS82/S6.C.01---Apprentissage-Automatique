from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from src.data.load_data import load_ml_datasets
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf

ML_DATA_FOLDER = Path("../../../data/ml")
MODELS_FOLDER = Path("../../../models")
MAX_LEN = 250
SAMPLE_SIZE = 1000


def train_bert_rating(print_eval: bool = False):
    df = load_ml_datasets(ML_DATA_FOLDER, "ml_reviews_rating.csv")

    df = df.sample(n=SAMPLE_SIZE, random_state=42)

    X = df["text"].values
    y = df["rating"].values - 1  # Soustraction de 1 pour que les classes soient 0, 1, 2, 3, 4 au lieu de 1, 2, 3, 4, 5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(
        list(X_train),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    test_encodings = tokenizer(
        list(X_test),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).shuffle(10000).batch(16)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(16)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=5
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(
        train_dataset,
        epochs=3,
        validation_data=test_dataset,
        verbose=print_eval
    )

    model.save_pretrained(MODELS_FOLDER / "model_bert_rating")
    tokenizer.save_pretrained(MODELS_FOLDER / "bert_rating_model_tokenizer")

    if print_eval:
        print("\nBERT rating predictions :")
        model.evaluate(test_dataset)


if __name__ == "__main__":
    train_bert_rating(print_eval=False)

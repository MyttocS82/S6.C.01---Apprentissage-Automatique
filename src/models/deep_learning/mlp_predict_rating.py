from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Dropout, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from src.data.load_data import load_ml_datasets
from pathlib import Path
import tensorflow as tf

ML_DATA_FOLDER = Path("../../../data/ml")
MODELS_FOLDER = Path("../../../models")

VOCAB_SIZE = 25000
MAX_LEN = 250


def custom_standardization(input_string):
    text = tf.strings.lower(input_string)
    text = tf.strings.regex_replace(text, r"[^\w\s]", "")
    return text


def train_mlp_rating(print_eval: bool = False):
    df = load_ml_datasets(ML_DATA_FOLDER, "ml_reviews_rating.csv")

    X = df["text"].values
    y = df["rating"].values - 1  # Soustraction de 1 pour que les classes soient 0, 1, 2, 3, 4 au lieu de 1, 2, 3, 4, 5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN,
        standardize=custom_standardization
    )
    vectorizer.adapt(X_train)

    model = tf.keras.Sequential([
        vectorizer,
        Embedding(VOCAB_SIZE, 128),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(5, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=1024,
        validation_split=0.1,
        verbose=print_eval  # Pour afficher ou non les détails de l'entraînement (perte, précision, etc.)
    )

    model.save(MODELS_FOLDER / "model_mlp_rating.keras")

    if print_eval:
        print("\nMLP rating predictions :")
        model.evaluate(X_test, y_test)


if __name__ == "__main__":
    train_mlp_rating(print_eval=True)

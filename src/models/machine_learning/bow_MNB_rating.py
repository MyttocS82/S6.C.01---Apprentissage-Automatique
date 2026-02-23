from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from src.data.load_data import load_ml_datasets
from pathlib import Path
import joblib

ML_DATA_FOLDER = Path("../../../data/ml")
MODELS_FOLDER = Path("../../../models")


def bow_mnb_rating(print_eval: bool = False) -> None:
    df = load_ml_datasets(ML_DATA_FOLDER, "ml_reviews_rating.csv")

    X = df["text"]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer(
        max_features=10000,
        lowercase=True
    )
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_bow, y_train)

    joblib.dump(model, MODELS_FOLDER / "model_bow_nb_rating.pkl")
    joblib.dump(vectorizer, MODELS_FOLDER / "vectorizer_bow_nb_rating.pkl")

    if print_eval:
        y_pred = model.predict(X_test_bow)

        print("\nBag-of-Words + Multinomial Naive Bayes (Rating)")
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average="weighted"))
        print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
        print("F1-score :", f1_score(y_test, y_pred, average="weighted"))


if __name__ == "__main__":
    bow_mnb_rating(print_eval=False)

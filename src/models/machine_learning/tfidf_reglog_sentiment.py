# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data.load_data import load_ml_datasets
from pathlib import Path
import joblib

ML_DATA_FOLDER = Path("../../../data/ml")
MODELS_FOLDER = Path("../../../models")

def tfidf_logisticregression_sentiment():
    # Chargement des données
    df_sentiment = load_ml_datasets(ML_DATA_FOLDER, "ml_reviews_sentiment.csv")

    X = df_sentiment["text"]
    y = df_sentiment["sentiment"]

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Regression logistique
    model = LogisticRegression(max_iter=2500)
    model.fit(X_train_tfidf, y_train)

    # Sauvegarde du modèle et du vectorizer
    joblib.dump(model, MODELS_FOLDER / 'model_sentiment.pkl')
    joblib.dump(vectorizer, MODELS_FOLDER / 'vectorizer_sentiment.pkl')

    '''
    # Prédiction
    y_test_pred = model.predict(X_test_tfidf)
    
    # Évaluation
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print("Accuracy : ", test_accuracy)
    print("Precision : ", test_precision)
    print("Recall : ", test_recall)
    print("F1-score : ", test_f1)
    '''

if __name__ == "__main__":
    tfidf_logisticregression_sentiment()

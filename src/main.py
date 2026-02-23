import os
import sys
import importlib.util
import pickle

import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
SRC_DIR = os.path.dirname(__file__)

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)


# ── helpers ────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def pick(options: list[str], prompt: str = "Votre choix") -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{prompt} (1-{len(options)}) : ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        print("  ⚠  Entrée invalide, veuillez réessayer.")


# ── IA methods ─────────────────────────────────────────────────────────────────
IA_MODULES = {
    "Zero-shot classification": ("classification_zero_shot", "zero_shot_predict_sentiment"),
    "Few-shot classification": ("classification_few_shot", "few_shot_predict_sentiment"),
}


def load_ia_module(module_name: str):
    path = os.path.join(SRC_DIR, "ia", f"{module_name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module introuvable : {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ia_on_text():
    print("\n── Méthode IA ──────────────────────────────────────────────────")
    names = list(IA_MODULES.keys())
    choice = pick(names, "Choisissez la méthode")

    display_name = names[choice - 1]
    module_filename, func_name = IA_MODULES[display_name]

    print(f"\n  Chargement de « {display_name} »…")
    try:
        module = load_ia_module(module_filename)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    if not hasattr(module, func_name):
        print(f"  ✗ Le module « {module_filename} » n'expose pas de fonction {func_name}().")
        return

    predict_fn = getattr(module, func_name)

    text = input("\nEntrez le texte à analyser : ").strip()
    if not text:
        print("  ⚠  Texte vide, abandon.")
        return

    print("\n  Analyse en cours…")
    result = predict_fn(text)
    print(f"\n  ✔  Résultat : {result}")


# ── model-based predictions on CSV ────────────────────────────────────────────

def custom_standardization(input_data):
    """
    Redéfinition de la fonction custom_standardization utilisée lors de l'entraînement
    du modèle CNN. Doit correspondre exactement à la fonction d'origine.
    """
    import tensorflow as tf
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, r"[^a-zA-Z0-9\s]", "")
    return stripped


def list_models() -> list[str]:
    entries = []
    for name in sorted(os.listdir(MODELS_DIR)):
        full = os.path.join(MODELS_DIR, name)
        # Exclure les vectorizers de la liste des modèles
        if name.startswith("vectorizer_"):
            continue
        if os.path.isfile(full) and name.endswith((".pkl", ".keras", ".h5")):
            entries.append(name)
        elif os.path.isdir(full) and not name.endswith("_tokenizer") and not name.endswith("_model_tokenizer"):
            entries.append(f"{name}/  (BERT)")
    return entries


def load_model(model_name: str):
    path = os.path.join(MODELS_DIR, model_name)

    if model_name.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f), "sklearn"

    if model_name.endswith(".keras") or model_name.endswith(".h5"):
        import tensorflow as tf
        import keras

        @keras.saving.register_keras_serializable()
        def custom_standardization_registered(input_data):
            lowercase = tf.strings.lower(input_data)
            stripped = tf.strings.regex_replace(lowercase, r"[^a-zA-Z0-9\s]", "")
            return stripped

        model = keras.models.load_model(
            path,
            custom_objects={"custom_standardization": custom_standardization_registered}
        )
        return model, "keras"

    if os.path.isdir(path):
        from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

        model = TFAutoModelForSequenceClassification.from_pretrained(path)

        # Détection du type : rating ou sentiment
        if "rating" in model_name:
            tokenizer_folder = "bert_rating_model_tokenizer"
        elif "sentiment" in model_name:
            tokenizer_folder = "bert_sentiment_model_tokenizer"
        else:
            tokenizer_folder = None

        # Priorité 1 : dossier bert_*_model_tokenizer/
        if tokenizer_folder:
            tokenizer_path = os.path.join(MODELS_DIR, tokenizer_folder)
            if os.path.isdir(tokenizer_path):
                print(f"  ↻ Tokenizer chargé depuis : {tokenizer_folder}/")
            else:
                tokenizer_path = None
        else:
            tokenizer_path = None

        # Priorité 2 : convention _tokenizer (ex: model_bert_rating_tokenizer/)
        if tokenizer_path is None:
            fallback = os.path.join(MODELS_DIR, f"{model_name}_tokenizer")
            if os.path.isdir(fallback):
                tokenizer_path = fallback
                print(f"  ↻ Tokenizer chargé depuis : {model_name}_tokenizer/")

        # Priorité 3 : même dossier que le modèle
        if tokenizer_path is None:
            tokenizer_path = path
            print(f"  ↻ Tokenizer chargé depuis le dossier du modèle.")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return (model, tokenizer), "bert"

    raise ValueError(f"Format de modèle non reconnu : {model_name}")


def find_vectorizer(model_name: str):
    base = model_name.replace("model_", "vectorizer_", 1)
    path = os.path.join(MODELS_DIR, base)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def predict_sklearn(model, vectorizer, texts: list[str]):
    X = vectorizer.transform(texts) if vectorizer else texts
    return model.predict(X)


def predict_keras(model, texts: list[str]):
    """
    Le modèle CNN intègre sa propre couche TextVectorization :
    il accepte directement des chaînes de caractères brutes.
    """
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices(texts).batch(256)
    all_preds = []
    for batch in dataset:
        preds = model.predict(batch, verbose=0)
        all_preds.append(preds)
    probabilities = np.concatenate(all_preds, axis=0)
    # Classe prédite (index 0-4 → rating 1-5)
    return np.argmax(probabilities, axis=1) + 1


def predict_bert(model_tok, texts: list[str]):
    model, tokenizer = model_tok
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
    logits = model(enc).logits.numpy()
    return np.argmax(logits, axis=1)


def decode_predictions(predictions, model_name: str) -> list:
    """
    Convertit les prédictions numériques en labels lisibles selon le type de modèle.
    - Modèles _sentiment : -1 (négatif), 0 (neutre), 1 (positif)
    - Modèles _rating    : 1 à 5 (étoiles)
    """
    if "sentiment" in model_name:
        mapping = {-1: "Négatif", 0: "Neutre", 1: "Positif"}
        return [mapping.get(int(p), str(p)) for p in predictions]
    if "rating" in model_name:
        return [f"{int(p)} étoile(s)" for p in predictions]
    return [str(p) for p in predictions]


def evaluate_predictions(df: pd.DataFrame, predictions, model_name: str, text_col: str) -> None:
    """
    Évalue les prédictions par rapport à une colonne de contrôle du CSV.
    Gère la conversion automatique si les échelles ne correspondent pas.
    """
    from src.utils.utils import create_label_review

    numeric_cols = [c for c in df.columns if c != text_col and c != "prediction" and c != "prediction_label"]
    if not numeric_cols:
        print("  ✗ Aucune colonne de contrôle disponible.")
        return

    print("\n  Colonnes disponibles pour l'évaluation :")
    idx = pick(numeric_cols, "Choisissez la colonne de contrôle")
    ctrl_col = numeric_cols[idx - 1]
    true_vals = df[ctrl_col].dropna()

    # ── Détection automatique de l'échelle de la colonne de contrôle ──────────
    unique_vals = sorted(true_vals.unique())
    print(f"\n  Valeurs uniques dans « {ctrl_col} » : {unique_vals}")

    true_labels = true_vals.copy()

    # Si la colonne de contrôle est en étoiles (1-5) et le modèle prédit du sentiment (-1,0,1)
    if "sentiment" in model_name and set(unique_vals).issubset({1, 2, 3, 4, 5}):
        print(f"  ↻ Conversion automatique des étoiles → sentiment (-1/0/1) pour la comparaison.")
        true_labels = true_vals.apply(create_label_review)

    # Si la colonne de contrôle est en sentiment (-1,0,1) et le modèle prédit du rating (1-5)
    elif "rating" in model_name and set(unique_vals).issubset({-1, 0, 1}):
        print(f"  ⚠  La colonne de contrôle est en sentiment mais le modèle prédit des ratings.")
        print(f"     La comparaison directe n'est pas possible.")
        return

    # Alignement sur l'index commun
    common_idx = true_labels.index
    preds_series = pd.Series(predictions, index=df.index).loc[common_idx]
    true_series = true_labels.loc[common_idx]

    # ── Métriques ─────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix
    )

    acc = accuracy_score(true_series, preds_series)
    report = classification_report(true_series, preds_series, zero_division=0)
    cm = confusion_matrix(true_series, preds_series)

    print(f"\n── Évaluation ──────────────────────────────────────────────────")
    print(f"  Modèle         : {model_name}")
    print(f"  Colonne réelle : {ctrl_col}")
    print(f"  Exemples       : {len(true_series)}")
    print(f"\n  ✔  Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"\n  Rapport de classification :\n")
    for line in report.splitlines():
        print(f"    {line}")
    print(f"\n  Matrice de confusion :")
    labels = sorted(true_series.unique())
    header_cm = "       " + "  ".join(f"{str(l):>6}" for l in labels)
    print(f"    {header_cm}")
    for i, label in enumerate(labels):
        row_str = "  ".join(f"{cm[i][j]:>6}" for j in range(len(labels)))
        print(f"    {str(label):>5}  {row_str}")


def run_model_on_csv():
    print("\n── Modèles disponibles ─────────────────────────────────────────")
    models = list_models()
    if not models:
        print("  ✗ Aucun modèle trouvé dans models/")
        return
    choice = pick(models, "Choisissez le modèle")
    model_name = models[choice - 1]

    csv_path = input("\nChemin vers le fichier CSV : ").strip().strip('"')
    if not os.path.isfile(csv_path):
        print(f"  ✗ Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n  Fichier chargé — {len(df)} lignes, colonnes : {list(df.columns)}")

    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        print("  ✗ Aucune colonne textuelle détectée.")
        return

    if len(text_cols) == 1:
        text_col = text_cols[0]
        print(f"  → Colonne textuelle détectée : « {text_col} »")
    else:
        print("\n  Colonnes textuelles disponibles :")
        idx = pick(text_cols, "Choisissez la colonne de texte")
        text_col = text_cols[idx - 1]

    texts = df[text_col].fillna("").tolist()
    clean_name = model_name.replace("/  (BERT)", "").strip()

    print(f"\n  Chargement du modèle « {clean_name} »…")
    try:
        loaded, kind = load_model(clean_name)
    except Exception as e:
        print(f"  ✗ Impossible de charger le modèle : {e}")
        return

    print("  Prédiction en cours…")
    try:
        if kind == "sklearn":
            vectorizer = find_vectorizer(clean_name)
            if vectorizer is None:
                print("  ⚠  Aucun vectorizer trouvé — les textes bruts seront utilisés.")
            predictions = predict_sklearn(loaded, vectorizer, texts)
        elif kind == "keras":
            predictions = predict_keras(loaded, texts)
        elif kind == "bert":
            predictions = predict_bert(loaded, texts)
        else:
            print("  ✗ Type de modèle inconnu.")
            return
    except Exception as e:
        print(f"  ✗ Erreur lors de la prédiction : {e}")
        return

    df["prediction"] = predictions
    df["prediction_label"] = decode_predictions(predictions, clean_name)

    # ── Affichage lisible ─────────────────────────────────────────────────────
    print("\n── Résultats ───────────────────────────────────────────────────")
    display_df = df[[text_col, "prediction", "prediction_label"]].copy()
    display_df[text_col] = display_df[text_col].str.slice(0, 80) + "…"

    col_w = 84
    pred_w = 12
    label_w = 20
    header = f"  {'Texte':<{col_w}} {'Prédit':>{pred_w}}  {'Label':<{label_w}}"
    sep = "  " + "-" * (col_w + pred_w + label_w + 4)
    print(header)
    print(sep)
    for _, row in display_df.head(20).iterrows():
        txt = str(row[text_col])[:col_w]
        pred = str(row["prediction"])
        label = str(row["prediction_label"])
        print(f"  {txt:<{col_w}} {pred:>{pred_w}}  {label:<{label_w}}")

    if len(df) > 20:
        print(f"\n  … {len(df) - 20} lignes supplémentaires non affichées.")

    # ── Actions post-prédiction ───────────────────────────────────────────────
    print("\n── Que souhaitez-vous faire ? ──────────────────────────────────")
    actions = [
        "Sauvegarder les résultats dans un CSV",
        "Évaluer les prédictions (comparer avec une colonne réelle)",
        "Les deux",
        "Rien",
    ]
    action = pick(actions, "Votre choix")

    if action in (1, 3):
        out = csv_path.replace(".csv", "_predictions.csv")
        df.to_csv(out, index=False)
        print(f"  ✔  Résultats sauvegardés dans : {out}")

    if action in (2, 3):
        evaluate_predictions(df, predictions, clean_name, text_col)


# ── main loop ──────────────────────────────────────────────────────────────────

def main():
    while True:
        print("\n╔══════════════════════════════════════╗")
        print("║   Outil de classification de texte   ║")
        print("╚══════════════════════════════════════╝")
        print("  1. Analyser une ligne de texte (méthode IA)")
        print("  2. Analyser un fichier CSV     (modèle entraîné)")
        print("  3. Quitter")

        raw = input("\nVotre choix (1-3) : ").strip()
        if raw == "1":
            run_ia_on_text()
        elif raw == "2":
            run_model_on_csv()
        elif raw == "3":
            print("\n  Au revoir !\n")
            break
        else:
            print("  ⚠  Entrée invalide.")


if __name__ == "__main__":
    main()

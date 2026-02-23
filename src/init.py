import os
import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
SRC_MODELS_DIR = ROOT_DIR / "src" / "models"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# ── Correspondance fichier source → (fichiers générés, nom de fonction, kwargs) ─
MODEL_TRAIN_MAP = {
    # machine_learning
    "bow_MNB_rating": (["model_bow_nb_rating.pkl", "vectorizer_bow_nb_rating.pkl"], "bow_mnb_rating",
                       {"print_eval": False}),
    "bow_MNB_sentiment": (["model_bow_nb_sentiment.pkl", "vectorizer_bow_nb_sentiment.pkl"], "bow_mnb_sentiment",
                          {"print_eval": False}),
    "tfidf_reglog_rating": (["model_reglog_rating.pkl", "vectorizer_reglog_rating.pkl"],
                            "tfidf_logisticregression_rating", {"print_eval": False}),
    "tfidf_reglog_sentiment": (["model_reglog_sentiment.pkl", "vectorizer_reglog_sentiment.pkl"],
                               "tfidf_logisticregression_sentiment", {"print_eval": False}),
    "tfidf_svm_rating": (["model_svc_rating.pkl", "vectorizer_svc_rating.pkl"], "tfidf_svm_rating",
                         {"print_eval": False}),
    "tfidf_svm_sentiment": (["model_svc_sentiment.pkl", "vectorizer_svc_sentiment.pkl"], "tfidf_svm_sentiment",
                            {"print_eval": False}),
    # deep_learning
    "mlp_predict_rating": (["model_mlp_rating.keras"], "mlp_predict_rating", {"print_eval": False}),
    "mlp_predict_sentiment": (["model_mlp_sentiment.keras"], "mlp_predict_sentiment", {"print_eval": False}),
    "cnn_predict_rating": (["model_cnn_rating.keras"], "cnn_predict_rating", {"print_eval": False}),
    "cnn_predict_sentiment": (["model_cnn_sentiment.keras"], "cnn_predict_sentiment", {"print_eval": False}),
    # transformers
    "bert_predict_rating": (["model_bert_rating/"], "bert_predict_rating", {"print_eval": False}),
    "bert_predict_sentiment": (["model_bert_sentiment/"], "bert_predict_sentiment", {"print_eval": False}),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def all_outputs_exist(outputs: list[str]) -> bool:
    for out in outputs:
        path = MODELS_DIR / out
        if out.endswith("/"):
            if not path.is_dir():
                return False
        else:
            if not path.is_file():
                return False
    return True


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_training_files() -> list[tuple[str, Path]]:
    """Retourne tous les fichiers .py des sous-dossiers de src/models/."""
    files = []
    for subdir in sorted(SRC_MODELS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for py_file in sorted(subdir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            files.append((py_file.stem, py_file))
    return files


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   Initialisation des modèles du projet   ║")
    print("╚══════════════════════════════════════════╝\n")

    MODELS_DIR.mkdir(exist_ok=True)

    training_files = collect_training_files()

    if not training_files:
        print("  ✗ Aucun fichier d'entraînement trouvé dans src/models/")
        return

    results = {"ok": [], "skipped": [], "error": []}

    for module_stem, file_path in training_files:
        entry = MODEL_TRAIN_MAP.get(module_stem)

        if entry is None:
            print(f"  ⚠  [{module_stem}] Non référencé dans MODEL_TRAIN_MAP — ignoré.")
            results["error"].append((module_stem, "Non référencé dans MODEL_TRAIN_MAP"))
            continue

        expected_outputs, fn_name, kwargs = entry

        # ── Vérification si le modèle existe déjà ─────────────────────────────
        if all_outputs_exist(expected_outputs):
            print(f"  ⏭  [{module_stem}] Déjà entraîné — ignoré.")
            results["skipped"].append(module_stem)
            continue

        print(f"\n  ▶  [{module_stem}] Entraînement en cours…")
        print(f"     Fichier : {file_path.relative_to(ROOT_DIR)}")
        print(f"     Fonction : {fn_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})")

        # Dans la boucle, avant train_fn(**kwargs) :
        try:
            module = load_module_from_path(module_stem, file_path)

            if not hasattr(module, fn_name):
                raise AttributeError(f"Fonction '{fn_name}' introuvable dans {file_path.name}")

            train_fn = getattr(module, fn_name)

            # Changer le répertoire de travail vers le dossier du fichier source
            original_cwd = os.getcwd()
            os.chdir(file_path.parent)
            try:
                train_fn(**kwargs)
            finally:
                os.chdir(original_cwd)  # Toujours restaurer, même en cas d'erreur

            print(f"  ✔  [{module_stem}] Terminé.")
            results["ok"].append(module_stem)

        except Exception as e:
            print(f"  ✗  [{module_stem}] Erreur : {e}")
            results["error"].append((module_stem, str(e)))

        # ── Résumé ─────────────────────────────────────────────────────────────────
        print("\n╔══════════════════════════════════╗")
        print("║            Résumé                ║")
        print("╚══════════════════════════════════╝")
        print(f"  ✔  Entraînés avec succès  : {len(results['ok'])}")
        print(f"  ⏭  Déjà présents (ignorés): {len(results['skipped'])}")
        print(f"  ✗  Erreurs                : {len(results['error'])}")

        if results["error"]:
            print("\n  Détail des erreurs :")
            for name, err in results["error"]:
                print(f"    - {name} : {err}")

        print("\n  Initialisation terminée.\n")


if __name__ == "__main__":
    main()

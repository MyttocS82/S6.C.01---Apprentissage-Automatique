# 🍽️ Analyse intelligente des avis Yelp

> Analyse de données, classification de sentiments et prédiction de notes à partir du dataset Yelp — avec Machine Learning, Deep Learning et IA générative.

---

## 📋 Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Structure du projet](#structure-du-projet)
- [Prérequis & Installation](#prérequis--installation)
- [Ordre d'exécution](#ordre-dexécution)
- [Partie A — Analyse de données](#partie-a--analyse-de-données)
- [Partie B — Modèles de prédiction](#partie-b--modèles-de-prédiction)
- [Utilisation de main.py](#utilisation-de-mainpy)
- [Modèles disponibles](#modèles-disponibles)
- [IA Générative](#ia-générative)

---

## 📌 Présentation du projet

Ce projet porte sur l'analyse et la classification automatique des avis de la plateforme **Yelp**. Il comprend deux grandes parties :

- **Partie A** : Analyse exploratoire des données (EDA) à partir des fichiers `business`, `users`, `reviews` et `photos`.
- **Partie B** : Modèles de prédiction de la **polarité** (positif / neutre / négatif) et du **score** (1 à 5 étoiles) des avis, en combinant plusieurs représentations textuelles et méthodes d'apprentissage.

---

## 🗂️ Structure du projet

```
.
├── data/
│   └── raw/                        # Fichiers JSON bruts du dataset Yelp
│       ├── yelp_academic_dataset_business.json
│       ├── yelp_academic_dataset_user.json
│       ├── yelp_academic_dataset_review.json
│       └── yelp_academic_dataset_photo.json
│
├── models/                         # Modèles entraînés (générés par init.py)
│   ├── model_bow_nb_rating.pkl
│   ├── vectorizer_bow_nb_rating.pkl
│   ├── model_reglog_rating.pkl
│   ├── vectorizer_reglog_rating.pkl
│   ├── model_svc_rating.pkl
│   ├── vectorizer_svc_rating.pkl
│   ├── model_mlp_rating.keras
│   ├── model_cnn_rating.keras
│   ├── model_bert_rating/          # Modèle BERT fine-tuné (rating)
│   ├── model_bert_sentiment/       # Modèle BERT fine-tuné (sentiment)
│   └── ...
│
├── results/
│   └── figures/                    # Graphiques générés par data_visualizations.py
│
├── src/
│   ├── data/
│   │   └── load_data.py            # Chargement des datasets Yelp
│   ├── ia/
│   │   ├── classification_zero_shot.py   # Classification zero-shot (LLM)
│   │   └── classification_few_shot.py    # Classification few-shot (LLM)
│   ├── models/
│   │   ├── machine_learning/       # Modèles ML classiques
│   │   ├── deep_learning/          # Modèles MLP & CNN
│   │   └── transformers/           # Modèles BERT
│   ├── utils/
│   │   └── utils.py                # Fonctions utilitaires (ex: create_label_review)
│   ├── visualization/
│   │   └── data_visualizations.py  # ⚠️ À exécuter en premier
│   ├── init.py                     # ⚠️ À exécuter en deuxième
│   └── main.py                     # Interface principale
```

---

## ⚙️ Prérequis & Installation

### Environnement Python recommandé

```bash
python >= 3.10
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

> Les principales librairies utilisées sont : `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` / `keras`, `transformers` (HuggingFace), `torch`.

---

## 🚀 Ordre d'exécution

> ⚠️ **Il est impératif de respecter l'ordre suivant avant d'utiliser `main.py`.**

### Étape 1 — Analyse exploratoire des données

```bash
python src/visualization/data_visualizations.py
```

Ce script charge les datasets Yelp et génère les **visualisations** dans `results/figures/`.

---

### Étape 2 — Entraînement des modèles

```bash
python src/init.py
```

Ce script parcourt tous les fichiers de `src/models/` et entraîne les modèles **qui ne sont pas encore présents** dans `models/`. Les modèles déjà générés sont automatiquement ignorés.

---

### Étape 3 — Interface de prédiction

```bash
python src/main.py
```

---

## 📊 Partie A — Analyse de données

Le script `data_visualizations.py` produit les analyses suivantes :

| # | Analyse | Visualisation |
|---|---------|---------------|
| 1 | Distribution des ratings dans le dataset reviews | Histogramme |
| 2 | Longueur des avis dans le dataset reviews | Histogramme |
| 3 | Longueur des avis par rapport aux notes | Boxplot |
| 4 | Lien entre le nombre d'avis d'un business et sa note moyenne | Scatter plot (échelle log) |
| 5 | Notes moyennes par rapport au nombre d'avis des utilisateurs | Scatter plot (échelle log) |
| 6 | Longueur des avis : utilisateurs expérimentés vs tous les utilisateurs | Boxplot |
| 7 | Longueur moyenne des reviews par classe de note (1 → 5) | Bar chart |

> Les figures sont sauvegardées automatiquement dans `results/figures/` au format PNG (résolution 500 dpi).

### Règle de labellisation de la polarité

| Score | Label |
|-------|-------|
| > 3   | ✅ Positif |
| = 3   | 😐 Neutre  |
| < 3   | ❌ Négatif |

---

## 🤖 Partie B — Modèles de prédiction

### Tâches

| Tâche | Description | Sortie |
|-------|-------------|--------|
| **Prédiction de polarité** | Classifier un avis en positif / neutre / négatif | `-1`, `0`, `1` |
| **Prédiction du score** | Prédire la note attribuée (1 à 5 étoiles) | `1`, `2`, `3`, `4`, `5` |

---

### 1. Représentations textuelles

| Représentation | Description |
|----------------|-------------|
| **Bag-of-Words (BoW)** | Sac de mots simples |
| **TF-IDF** | Pondération terme-fréquence / fréquence inverse |
| **Embeddings BERT** | Représentations contextuelles issues de BERT pré-entraîné |

---

### 2. Méthodes d'apprentissage

| Catégorie | Modèles |
|-----------|---------|
| **Machine Learning classique** | Naive Bayes (BoW), Régression Logistique (TF-IDF), SVM (TF-IDF) |
| **Deep Learning** | MLP, CNN avec couche TextVectorization intégrée |
| **Transformers** | BERT fine-tuné (TFAutoModelForSequenceClassification) |

---

### 3. Conventions de nommage des fichiers modèles

```
model_<type>_<tâche>.pkl/.keras    →  modèle entraîné
vectorizer_<type>_<tâche>.pkl      →  vectorizer associé (BoW / TF-IDF)
model_bert_<tâche>/                →  dossier modèle BERT
bert_<tâche>_model_tokenizer/      →  dossier tokenizer BERT
```

**Exemples :**
```
model_bow_nb_rating.pkl
vectorizer_bow_nb_rating.pkl
model_cnn_sentiment.keras
model_bert_rating/
bert_rating_model_tokenizer/
```

---

## 🖥️ Utilisation de main.py

```
╔══════════════════════════════════════╗
║   Outil de classification de texte   ║
╚══════════════════════════════════════╝
  1. Analyser une ligne de texte (méthode IA)
  2. Analyser un fichier CSV     (modèle entraîné)
  3. Quitter
```

### Option 1 — Analyse d'un texte via IA générative

Permet d'analyser un texte en entrant directement une phrase ou une revue.  
Deux méthodes disponibles :
- **Zero-shot classification** : le LLM prédit directement le sentiment sans exemple.
- **Few-shot classification** : le LLM reçoit quelques exemples avant de prédire.

### Option 2 — Analyse d'un fichier CSV via un modèle entraîné

1. Sélectionner un modèle parmi ceux disponibles dans `models/`
2. Fournir le chemin vers un fichier CSV
3. Sélectionner la colonne textuelle à analyser
4. Les prédictions sont affichées (20 premières lignes) puis il est proposé de :
   - 💾 Sauvegarder les résultats dans un nouveau CSV (`*_predictions.csv`)
   - 📊 Évaluer les prédictions par rapport à une colonne réelle (accuracy, rapport de classification, matrice de confusion)
   - Les deux
   - Ne rien faire

---

## 🧠 IA Générative

### Zero-shot

```
src/ia/classification_zero_shot.py
Fonction : zero_shot_predict_sentiment(text: str) -> str
```

Le LLM reçoit uniquement le texte de la revue et produit une prédiction de sentiment **sans données d'entraînement**.

### Few-shot

```
src/ia/classification_few_shot.py
Fonction : few_shot_predict_sentiment(text: str) -> str
```

Le LLM reçoit quelques exemples annotés (positif / négatif / neutre) avant de prédire le sentiment du texte fourni.

### Aspect-Based Sentiment Analysis (ABSA)

Le LLM produit une **sortie structurée** identifiant :
- les aspects mentionnés dans la revue (ex: nourriture, service, prix)
- le sentiment associé à chacun d'eux (positif / négatif)

---

## 📁 Dataset Yelp

Ce projet utilise le [Yelp Open Dataset](https://www.yelp.com/dataset).

| Fichier | Contenu | Entrées |
|---------|---------|---------|
| `yelp_academic_dataset_business.json` | Informations sur les établissements | ~150 346 |
| `yelp_academic_dataset_user.json` | Profils utilisateurs | ~558 095 |
| `yelp_academic_dataset_review.json` | Avis textuels avec notes | ~1 000 000 |
| `yelp_academic_dataset_photo.json` | Photos associées aux établissements | — |

> ⚠️ Les fichiers de données ne sont **pas inclus** dans ce dépôt. Téléchargez-les depuis le site officiel Yelp et placez-les dans `data/raw/`.

---

## 👨‍💻 Auteur
**LAM Clément** //
**LE VELLY Malek** //
**MASSAT Diego** //
**MICHELON Scott**

Module : `S6.C.01 - Apprentissage Automatique` - BUT Informatique 3 (AGED)

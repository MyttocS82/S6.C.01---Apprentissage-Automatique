# S6.C.01 - Apprentissage Automatique
## Analyse intelligente des avis Yelp avec ML, Deep Learning et IA agentique

Ce projet s'appuie sur le **Yelp Open Dataset** pour développer des outils d'analyse et de machine learning appliqués aux données de la plateforme Yelp.

## 📋 Description

Yelp (https://www.yelp.com) est un service de recommandation en ligne qui permet aux utilisateurs de :
- **Rechercher des commerces locaux** (restaurants, hôtels, bars, coiffeurs, garages, etc.)
- **Consulter des avis** rédigés par d'autres clients
- **Noter ces établissements** avec un score de 1 à 5 étoiles
- **Publier leurs propres revues**, parfois accompagnées de photos

Ce projet propose des outils pour analyser ces données et développer des modèles de machine learning et de deep learning.

## 🚀 Fonctionnalités

### 1. Chargement de données
- Import des données Yelp (businesses, reviews, users)
- Support du format JSON du Yelp Open Dataset
- Chargement optimisé avec limitation optionnelle

### 2. Recherche d'établissements
- Recherche par nom d'établissement
- Filtrage par catégorie (restaurants, hôtels, etc.)
- Filtrage par note (1-5 étoiles)
- Filtrage par localisation (ville, état)
- Identification des établissements les mieux notés

### 3. Analyse des avis
- Distribution des notes (1-5 étoiles)
- Analyse des avis par établissement
- Analyse des avis par utilisateur
- Statistiques sur les textes des avis
- Identification des avis les plus utiles
- Analyse de sentiment par note

### 4. Visualisations
- Distribution des étoiles
- Top catégories d'établissements
- Relation entre nombre d'avis et note moyenne
- Évolution des avis au fil du temps

## 📦 Installation

```bash
# Cloner le dépôt
git clone https://github.com/MyttocS82/S6.C.01---Apprentissage-Automatique.git
cd S6.C.01---Apprentissage-Automatique

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Dataset Yelp

### Téléchargement
Téléchargez le Yelp Open Dataset depuis : https://www.yelp.com/dataset

Le dataset contient plusieurs fichiers JSON :
- `yelp_academic_dataset_business.json` - Informations sur les établissements
- `yelp_academic_dataset_review.json` - Avis des utilisateurs
- `yelp_academic_dataset_user.json` - Informations sur les utilisateurs

### Structure des données

**Business (Établissements)**
- `business_id` : Identifiant unique
- `name` : Nom de l'établissement
- `stars` : Note moyenne (1-5)
- `review_count` : Nombre d'avis
- `categories` : Catégories (restaurants, hôtels, etc.)
- `city`, `state` : Localisation

**Reviews (Avis)**
- `review_id` : Identifiant unique
- `user_id` : Identifiant de l'utilisateur
- `business_id` : Identifiant de l'établissement
- `stars` : Note donnée (1-5)
- `text` : Texte de l'avis
- `date` : Date de publication
- `useful`, `funny`, `cool` : Votes des autres utilisateurs

## 💻 Utilisation

### Exemple de base

```python
from pathlib import Path
from yelp_analysis.data_loader import YelpDataLoader
from yelp_analysis.business_search import BusinessSearch
from yelp_analysis.review_analyzer import ReviewAnalyzer

# Charger les données
data_dir = Path("data/raw")
loader = YelpDataLoader(data_dir)

# Charger les établissements (limité à 1000 pour l'exemple)
businesses = loader.load_businesses(limit=1000)

# Rechercher des restaurants
search = BusinessSearch(businesses)
restaurants = search.search_by_category("Restaurants")
top_restaurants = search.get_top_rated(n=10, category="Restaurants")

# Charger et analyser les avis
reviews = loader.load_reviews(limit=5000)
analyzer = ReviewAnalyzer(reviews)

# Distribution des notes
distribution = analyzer.get_rating_distribution()
print(distribution)

# Note moyenne
avg_rating = analyzer.get_average_rating()
print(f"Note moyenne: {avg_rating:.2f} étoiles")
```

### Notebooks d'exemple

Consultez le répertoire `notebooks/` pour des exemples complets :
- `example_analysis.ipynb` - Analyse exploratoire des données

## 🏗️ Structure du projet

```
S6.C.01---Apprentissage-Automatique/
├── src/
│   └── yelp_analysis/          # Package principal
│       ├── __init__.py
│       ├── config.py           # Configuration
│       ├── data_loader.py      # Chargement des données
│       ├── business_search.py  # Recherche d'établissements
│       ├── review_analyzer.py  # Analyse des avis
│       └── visualizations.py   # Visualisations
├── data/
│   ├── raw/                    # Données brutes Yelp
│   └── processed/              # Données traitées
├── notebooks/                  # Notebooks Jupyter
├── tests/                      # Tests unitaires
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🔬 Développements futurs

- Modèles de classification de sentiment
- Recommandation d'établissements
- Détection d'avis frauduleux
- Analyse de texte avec NLP avancé
- Modèles de deep learning (BERT, GPT)
- Interface web interactive

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📧 Contact

Pour toute question, contactez l'équipe du projet.

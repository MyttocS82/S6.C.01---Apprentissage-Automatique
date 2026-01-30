# Data Directory

Ce répertoire contient les données du Yelp Open Dataset.

## Structure

```
data/
├── raw/                # Données brutes du Yelp Open Dataset
└── processed/          # Données traitées et préparées pour l'analyse
```

## Téléchargement du Dataset

Pour utiliser ce projet, vous devez télécharger le **Yelp Open Dataset** depuis :
https://www.yelp.com/dataset

### Fichiers nécessaires

Téléchargez et placez les fichiers suivants dans le répertoire `data/raw/` :

1. **yelp_academic_dataset_business.json**
   - Informations sur les établissements (restaurants, hôtels, etc.)
   - Contient : nom, adresse, catégories, notes, etc.

2. **yelp_academic_dataset_review.json**
   - Avis rédigés par les utilisateurs
   - Contient : texte de l'avis, note (1-5 étoiles), date, votes utiles, etc.

3. **yelp_academic_dataset_user.json**
   - Informations sur les utilisateurs
   - Contient : nombre d'avis, amis, votes reçus, etc.

## Format des données

Les fichiers sont au format **JSON Lines** (un objet JSON par ligne).

### Exemple Business
```json
{
  "business_id": "abc123",
  "name": "Restaurant Exemple",
  "address": "123 Main St",
  "city": "Phoenix",
  "state": "AZ",
  "postal_code": "85001",
  "latitude": 33.4484,
  "longitude": -112.0740,
  "stars": 4.5,
  "review_count": 150,
  "categories": "Restaurants, Italian, Pizza"
}
```

### Exemple Review
```json
{
  "review_id": "xyz789",
  "user_id": "user123",
  "business_id": "abc123",
  "stars": 5,
  "text": "Excellent restaurant! La nourriture était délicieuse...",
  "date": "2024-01-15",
  "useful": 12,
  "funny": 2,
  "cool": 5
}
```

## Note importante

Les fichiers de données ne sont pas inclus dans le dépôt Git en raison de leur taille importante.
Vous devez les télécharger manuellement depuis le site de Yelp.

from src.data.load_data import load_yelp_datasets
from matplotlib import pyplot as plt
from pathlib import Path

raw_data_folder = Path("../../data/raw")
buisness, users, reviews = load_yelp_datasets(raw_data_folder)

print("Buisness DataFrame Info:", buisness.info(), "\n")
print("Users DataFrame Info:", users.info(), "\n")
print("Reviews DataFrame Info:", reviews.info(), "\n")
"""
Conclusion from DataFrame Info:
* Buisness (150346 entries, 14 columns):
    3 colonnes ont des valeurs absentes ou nulles : 11. attributes(13 744), 12. categories(103), 13. hours(23 223)
    Les colonnes suivantes sont de type 'object' et peuvent contenir des données textuelles ou catégorielles :
        - business_id, name, address, city, state, postal_code, attributes, categories, hours
        
* Users (558095 entries, 22 columns):
    Pas de valeurs absentes ou nulles.
    Les colonnes suivantes sont de type 'object' et peuvent contenir des données textuelles ou catégorielles :
        - user_id, name, yelping_since, elite, friends (user_id d'autres personnes en liste)
        
* Reviews (1000000 entries, 9 columns):
    Pas de valeurs absentes ou nulles.
    Les colonnes suivantes sont de type 'object' et peuvent contenir des données textuelles ou catégorielles :
        - review_id, user_id, business_id, text, date(datetime64[ms])
"""
# Ajout d'une colonne pour la longueur des avis
reviews['text_length'] = reviews['text'].apply(len)

# Visualisation 1 : distribution des ratings dans le dataset reviews
plt.hist(reviews['stars'], bins=5)
plt.title('Distribution des ratings dans le dataset reviews')
plt.xlabel('Nb Étoiles')
plt.ylabel('Nombre de Reviews')
plt.xticks([1, 2, 3, 4, 5])
plt.savefig("../../results/figures/Distribution des ratings dans le dataset reviews.png", dpi=500)
plt.show()

# Visualisation 2 : longueur des avis dans le dataset reviews
plt.hist(reviews['text_length'], bins=50)
plt.title('Longueur des avis dans le dataset reviews')
plt.xlabel('Longueur de l\'avis (nombre de caractères)')
plt.ylabel('Nombre de Reviews')
plt.savefig("../../results/figures/Longueur des avis dans le dataset reviews.png", dpi=500)
plt.show()

# Visualisation 3 : longueur des avis par rapport aux notes (boxplot)
reviews.boxplot(column='text_length', by='stars')
plt.title('Longueur des avis par rapport aux notes')
plt.xlabel('Nb Étoiles')
plt.ylabel('Longueur de l\'avis (nombre de caractères)')
plt.savefig("../../results/figures/Longueur des avis par rapport aux notes.png", dpi=500)
plt.show()

# Visualisation 4 : lien entre le nombre total d’avis d’un business et la note moyenne du business
business_review_stats = reviews.groupby('business_id').agg({'stars': ['mean', 'count']})
business_review_stats.columns = ['average_stars', 'review_count']
plt.scatter(business_review_stats['review_count'], business_review_stats['average_stars'], alpha=0.5)
plt.title('Lien entre le nombre total d’avis d’un business et la note moyenne du business')
plt.xlabel('Nombre total d\'avis')
plt.ylabel('Note moyenne du business')
plt.xscale('log')       # Voir si on garde le log ou pas !
plt.yscale('linear')
plt.savefig("../../results/figures/Lien entre le nombre total d’avis d’un business et la note moyenne du business.png", dpi=500)
plt.show()

# Visualisation 5 : notes par rapports aux nombres d'avis des utilisateurs
user_review_stats = reviews.groupby('user_id').agg({'stars': ['mean', 'count']})
user_review_stats.columns = ['average_stars', 'review_count']
plt.scatter(user_review_stats['review_count'], user_review_stats['average_stars'], alpha=0.5, color='orange')
plt.title('Notes par rapports aux nombres d\'avis des utilisateurs')
plt.xlabel('Nombre total d\'avis par utilisateur')
plt.ylabel('Note moyenne par utilisateur')
plt.xscale('log')       # Voir si on garde le log ou pas !
plt.yscale('linear')
plt.savefig("../../results/figures/Notes par rapports aux nombres d'avis des utilisateurs.png", dpi=500)
plt.show()

# Visualisation 6 : TODO (Est-ce que les utilisateurs expérimentés, ont tendance à faire des reviews plus détaillées ?)

# Visualisation 7 : longueur moyenne des reviews par classe de note
average_length_by_stars = reviews.groupby('stars')['text_length'].mean()
average_length_by_stars.plot(kind='bar', color='green')
plt.title('Longueur moyenne des reviews par classe de note')
plt.xlabel('Nb Étoiles')
plt.ylabel('Longueur moyenne de l\'avis (nombre de caractères)')
plt.savefig("../../results/figures/Longueur moyenne des reviews par classe de note.png", dpi=500)
plt.show()

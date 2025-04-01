import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Chemin vers les fichiers de données
text_data_path = 'X_train_cleaned.csv'

# Chargement des données textuelles
text_data = pd.read_csv(text_data_path,index_col=0)

# Affichage des premières lignes des données textuelles
print(text_data.head())

# Affichage des informations sur les données textuelles
print(text_data.info())

# Visualisation de la répartition des classes
class_distribution = text_data['product_type_code'].value_counts()
print(class_distribution)
# Nuage de mots pour les descriptions

descriptions = text_data['cleaned_description'].dropna().values
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(descriptions))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
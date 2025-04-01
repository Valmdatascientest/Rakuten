import pandas as pd
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Chargement des données textuelles
text_data = pd.read_csv('X_train_update.csv', index_col=0)
text_data['product_type_code'] = pd.read_csv('Y_train_CVw08PX.csv', index_col=0)

# Téléchargement des ressources NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')

# Fonction de nettoyage des textes
def clean_text(text):
    # Convertir en minuscules
    text = text.lower()

    # Supprimer les balises HTML
    text = re.sub(r'<.*?>', '', text)

    # Décoder les caractères HTML
    text = html.unescape(text)

    # Supprimer les URL
    text = re.sub(r'http\S+', '', text)

    # Décoder les caractères HTML
    text = html.unescape(text)

    # Supprimer les caractères de ponctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)

    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Supprimer les espaces en début et fin de texte
    text = text.strip()
    
    # Supprimer les caractères spéciaux sauf les lettres accentuées et les espaces
    text = re.sub(r'[^a-zA-Zàâôéèêç\s]', '', text)

    # Tokenisation
    tokens = word_tokenize(text)
    # Supprimer les stopwords
    tokens = [word for word in tokens if word not in stopwords.words('french')]
    return ' '.join(tokens)

# Application du nettoyage aux descriptions
text_data.dropna(subset=['description'], inplace=True)
text_data['cleaned_description'] = text_data['description'].apply(clean_text)
#text_data.drop(columns=['description'], inplace=True)
text_data.to_csv('X_train_cleaned.csv')


# Affichage des premières lignes des descriptions nettoyées
print(text_data[['description', 'cleaned_description']].head(20))
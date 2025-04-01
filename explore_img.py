import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Chemin vers les fichiers de données
data_path = 'X_train_update.csv'
product_type_code_path = 'Y_train_CVw08PX.csv'
image_data_path = 'images/image_train'

# Chargement des données textuelles
text_data = pd.read_csv(data_path,index_col=0)
text_data['product_type_code'] = pd.read_csv(product_type_code_path, index_col=0)
# création d'une colonne avec le nom des images
text_data['nom_image'] = text_data.apply(lambda row: f"{image_data_path}/image_{row['imageid']}_product_{row['productid']}.jpg", axis=1)
# creation d'une colonne avec le label de product type code
text_data['label'] = text_data['product_type_code'].replace({
    10 : 1,#"Livre occasion",
        40 : 2,#"Jeu vidéo, accessoire tech.",
        50 : 3,#"Accessoire Console",
        60 : 4,# "Console de jeu",
        1140 : 5,# "Figurine",
        1160 : 6,#"Carte Collection",
        1180 : 7,# "Jeu Plateau",
        1280 : 8,#"Jouet enfant, déguisement",
        1281 : 9,#"Jeu de société",
        1300 : 10,#"Jouet tech",
        1301 : 11,#"Paire de chaussettes",
        1302 : 12,#"Jeu extérieur, vêtement",
        1320 : 13,#"Autour du bébé",
        1560 : 14,#"Mobilier intérieur",
        1920 : 15,#"Chambre",
        1940 : 16,#"Cuisine",
        2060 : 17,#"Décoration intérieure",
        2220 : 18,#"Animal",
        2280 : 19,#"Revues et journaux",
        2403 : 20,# "Magazines, livres et BDs",
        2462 : 21,#"Jeu occasion",
        2522 : 22,#"Bureautique et papeterie",
        2582 : 23,#"Mobilier extérieur",
        2583 : 24,#"Autour de la piscine",
        2585 : 25,#"Bricolage",
        2705 : 26,#"Livre neuf",
        2905 : 27,#"Jeu PC",
})
# Chargement des données d'images
# Note : les images sont stockées dans un dossier séparé
# et sont référencées par leur nom dans le fichier CSV
print(text_data['nom_image'].head(), text_data['label'].head())

# Affichage des premières lignes des données textuelles
print(text_data.head())

# Affichage des informations sur les données textuelles
print(text_data.info())

# Visualisation de la répartition des classes
class_distribution = text_data['label'].value_counts()
print(class_distribution)
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar')
plt.title('Distribution des classes de produits')
plt.xlabel('Classes de produits')
plt.ylabel('Nombre de produits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualisation des images
# Affichage de quelques images


plt.figure(figsize=(12, 8))
for i in range(6):  # Afficher 6 images par exemple
    img = Image.open(text_data['nom_image'].iloc[i])
    img = img.resize((500, 500))  # Redimensionner l'image pour un affichage uniforme
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(text_data['label'].iloc[i])
    plt.axis('off')
plt.show()

# Statistiques descriptives
print("Nombre total de produits :", len(text_data))
print("Nombre de classes :", len(class_distribution))
print("Longueur moyenne des descriptions :", text_data['description'].str.len().mean())
text_data['description'] = text_data['description'].astype(str)
print("Longueur maximale des descriptions :", text_data['description'].str.len().max())
print("Longueur minimale des descriptions :", text_data['description'].str.len().min())

text_data.to_csv('dataimg_label.csv', index=False)

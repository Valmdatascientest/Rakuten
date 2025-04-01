import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
import cv2
import random
import keras
import keras_hub
from keras.preprocessing import image
from keras.applications import imagenet_utils
from tensorflow.keras.utils import to_categorical

# Définir le modèle en utilisant l'approche fonctionnelle

def create_model():
    """input_layer = layers.Input(shape=(256, 256, 3), name='input')

    # Appliquer les couches d'augmentation de données
    x = data_augmentation()

    # Normalisation des valeurs des pixels
    x = layers.Rescaling(1.0 / 255)(x)

    # Redimensionnement des images
    x = layers.Resizing(64, 64)(x)

    # Première couche de convolution
    x = layers.Conv2D(32, (4, 4), activation='relu', name='conv2d_1')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)

    # Deuxième couche de convolution
    x = layers.Conv2D(32, (4, 4), activation='relu', name='conv2d_2')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_2')(x)

    # Aplatir les données pour les couches denses
    x = layers.Flatten(name='flatten')(x)

    # Première couche dense
    x = layers.Dense(1, activation='relu', name='dense_1')(x)

    # Dernière couche dense avec activation sigmoid
    output_layer = layers.Dense(28, activation='sigmoid', name='dense_2')(x)

    # Créer le modèle
    model = models.Model(inputs=input_layer, outputs=output_layer)"""
    # Initialisation du modèle séquentiel
    model = Sequential()

    # Ajout de couches de convolution et de pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
    model.add(MaxPooling2D((2, 2)))

    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))

   #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    
    # Utilisation de GlobalAveragePooling2D pour réduire la dimensionnalité
    model.add(GlobalAveragePooling2D())

    # Aplatissement des données pour les couches denses
    model.add(Flatten())

    # Ajout de couches denses
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='softmax'))
    return model

def data_augmentation():
    """Crée une couche d'augmentation de données."""
    return tf.keras.Sequential([
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2, name='random_translation'),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2, name='random_zoom'),
        layers.RandomFlip("horizontal", name='random_flip'),
        layers.RandomRotation(factor=0.2, name='random_rotation'),
        layers.RandomContrast(factor=0.2, name='random_contrast'),
    ])
# Fonction pour lire un fichier CSV et retourner un DataFrame
def lire_csv(fichier):
    """Lit un fichier CSV et retourne un DataFrame."""
    return pd.read_csv(fichier,index_col=0)

def analyser_donnees(df):
    """Effectue des analyses de base sur le DataFrame."""
    print("Aperçu des données :")
    print(df.head())

    print("\nStatistiques descriptives :")
    print(df.describe())

    print("\nTypes de données :")
    print(df.dtypes)

    print("Visualisation de Nan :")
    print(df.isnull().sum())

def visualiser_donnees(df):
    """Visualise les données avec des graphiques."""
    df.plot(kind='hist', bins=30, alpha=0.5)
    plt.title('Histogramme des données')
    plt.xlabel('Valeurs')
    plt.ylabel('Fréquence')
    plt.show()
def load_image(image_path, label):
    """Charge une image à partir d'un chemin et redimensionne l'image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    return image, label
def visualiser_distribution_classes(labels):
    """Visualise la distribution des classes dans les étiquettes."""
    unique_labels, counts = tf.unique_with_counts(labels)
    plt.bar(unique_labels.numpy(), counts.numpy())
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'exemples')
    plt.title('Distribution des classes')
    plt.show()
def visualiser_images(images, labels):
    """Visualise un échantillon d'images avec leurs étiquettes."""
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Afficher 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Convertir en uint8 pour l'affichage
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")  # Masquer les axes
    plt.show()
def visualiser_performance(history):
    """Visualise les performances du modèle pendant l'entraînement."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
def visualiser_predictions(images, predictions):
    """Visualise les prédictions du modèle sur un échantillon d'images."""
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Predicted: {predictions[i]}")
        plt.axis("off")
    plt.show()


def main():
    # Charger les données
   # fichier = 'X_train_update.csv'  # Remplacez par le chemin de votre fichier CSV
   # X_train = lire_csv(fichier)
   # Y_train = lire_csv('Y_train_CVw08PX.csv')
   # X_test = lire_csv('X_test_update.csv')
   # df = pd.concat([X_train,Y_train],axis=1)
    #analyser_donnees(df)
    fichier = 'dataimg_label.csv'  # Remplacez par le chemin de votre fichier CSV
    df = lire_csv(fichier)
    data_ds_img_path = df["nom_image"]
    labels_ds =to_categorical(df["label"], num_classes=28)
    # Créer un ensemble de données à partir des chemins d'images et des étiquettes
    data_ds = tf.data.Dataset.from_tensor_slices((data_ds_img_path, labels_ds))
    

    # Appliquer la fonction de chargement d'image
    data_ds = data_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Diviser les données en ensembles d'entraînement et de validation
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_ds = data_ds.take(train_size)
    val_ds = data_ds.skip(train_size)
    # Prétraiter les données
    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    # Normaliser les images
    def normalize(image, label):
        image = tf.image.per_image_standardization(image)
        return image, label
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2, name='random_translation'),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2, name='random_zoom'),
        layers.RandomFlip("horizontal", name='random_flip'),
        layers.RandomRotation(factor=0.2, name='random_rotation'),
        layers.RandomContrast(factor=0.2, name='random_contrast'),
    ])

    # Appliquer le prétraitement
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # Appliquer l'augmentation de données
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Préparer les ensembles de données pour l'entraînement
    batch_size = 32
    #train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    # Visualiser les données
    images, labels = next(iter(train_ds))
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Afficher 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Convertir en uint8 pour l'affichage
        plt.title(f"Label: {labels[i]}")
        plt.axis("off") # Masquer les axes
    plt.show()
    # Afficher les dimensions des images
    print(f"Dimensions des images : {images.shape}")
    print(f"Dimensions des étiquettes : {labels.shape}")
    # Afficher les types de données
    print(f"Type de données des images : {images.dtype}")
    print(f"Type de données des étiquettes : {labels.dtype}")
    # Afficher les valeurs uniques des étiquettes
    #print(f"Valeurs uniques des étiquettes : {tf.unique(labels)}")
    # Afficher la taille de l'ensemble de données
    print(f"Taille de l'ensemble d'entraînement : {len(train_ds)}")
    print(f"Taille de l'ensemble de validation : {len(val_ds)}")
    # Afficher la distribution des classes
    #visualiser_distribution_classes(labels)

    
    

    # Créer le modèle
    model = create_model()
    # Afficher le résumé du modèle
    model.summary()
    # Compiler et entraîner le modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Afficher la structure du modèle
    model.summary()
    # Afficher les poids du modèle
    for layer in model.layers:
        print(f"Poids de la couche {layer.name} : {layer.get_weights()}")
    # Afficher les poids de la couche de convolution
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            print(f"Poids de la couche {layer.name} : {layer.get_weights()}")
    # Afficher les poids de la couche dense
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            print(f"Poids de la couche {layer.name} : {layer.get_weights()}")
    # Afficher les poids de la couche de sortie
    for layer in model.layers:
        if isinstance(layer, layers.Dense) and layer.name == 'dense_2':
            print(f"Poids de la couche {layer.name} : {layer.get_weights()}")
    # Afficher le nombre de paramètres du modèle
    total_params = model.count_params()
    print(f"Nombre total de paramètres : {total_params}")
    # Afficher le nombre de paramètres entraînables
    trainable_params = sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
    print(f"Nombre de paramètres entraînables : {trainable_params}")
    # Afficher le nombre de paramètres non entraînables
    non_trainable_params = total_params - trainable_params
    print(f"Nombre de paramètres non entraînables : {non_trainable_params}")
    # Afficher le nombre de paramètres par couche
    for layer in model.layers:
        layer_params = layer.count_params()
        print(f"Nombre de paramètres de la couche {layer.name} : {layer_params}")

    
    print("affichage DS",val_ds,train_ds)
    
    
    # Entraîner le modèle
    #model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Évaluer le modèle sur l'ensemble de validation
    #val_loss, val_accuracy = model.evaluate(val_ds)
    #print(f"Validation Loss: {val_loss}")
    #print(f"Validation Accuracy: {val_accuracy}")
    # Visualiser les performances du modèle
    history = model.fit(train_ds, validation_data=val_ds, epochs=20 )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    # Visualiser la perte
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    # Visualiser les prédictions
    predictions = model.predict(val_ds)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Predicted: {predictions[i]}")
        plt.axis("off")
    plt.show()
    # Enregistrer les poids du modèle
    model.save_weights('model.weights.h5')
    
    # Sauvegarder le modèle
    model.save('model.h5')


if __name__ == "__main__":
    main()

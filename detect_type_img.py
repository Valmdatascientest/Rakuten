import keras
import keras_hub
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications import imagenet_utils
import os



# Chemin vers l'image à tester
image_path="images/image_train/image_963551423_product_243785540.jpg"

# Charger le modèle Faster R-CNN pré-entraîné
model =  keras_hub.models.ImageClassifier.from_preset(
    "resnet_50_imagenet",
    activation="softmax",
)

# Fonction pour détecter les objets dans une image
def detect_objects(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)

    predictions = model.predict(img_array)
    return predictions



predictions = detect_objects(image_path, model)

# Afficher les prédictions
print(predictions)
snset = imagenet_utils.decode_predictions(predictions, top=5)[0]
for i, (imagenet_id, label, score) in enumerate(snset):
    print(f"{i + 1}: {label} ({score:.2f})")
# Afficher les classes détectées
detected_classes = []
for i, (imagenet_id, label, score) in enumerate(snset):
    detected_classes.append(label)
# Afficher les classes détectées
print("Classes détectées :")
for label in detected_classes:
    print(label)
# Afficher les classes détectées
print("Classes détectées :")
for label in detected_classes:
    print(label)
# 

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Charger ResNet50 pré-entraîné
base_model = ResNet50(weights='imagenet', include_top=False)

# Déverrouiller les 27 dernières couches
for layer in base_model.layers[-27:]:
    layer.trainable = True

# Ajouter des couches personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1000, activation='softmax')(x)  # Assurez-vous que le nombre de classes est correct

# Créer le modèle final
model_images = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
model_images.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher l'architecture du modèle
model_images.summary()
model_images.save('model_images.h5')

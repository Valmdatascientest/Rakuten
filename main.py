# Authors: Valentin MILLLIET
# Creation date: 27/03/2025
def main():
    #import img_resnet50
    #import textuel_cnn
    #import prepro_txt
    from tensorflow.keras.models import load_model
    import pandas as pd
    import numpy as np
    #from sklearn.model_selection import train_test_split


    # Chemin vers les fichier .h5
    model_text_path = 'model_text.h5'
    model_images_path = 'model_images.h5'

    # Charger les modèles
    model_text = load_model(model_text_path)
    #model_images= load_model(model_images_path)

    # Charger les données
    df_train = pd.read_csv('X_train_cleaned.csv',index_col=0)
    X_train = df_train.drop("product_type_code", axis=1)
    X_train = df_train.drop("product_type", axis=1)
    y_train = df_train["product_type_code"]
    #y_train = pd.read_csv('Y_train_CVw08PX.csv',index_col=0)
    #df_test = pd.read_csv('X_test_update.csv')
    #y_test = pd.read_csv('Y_train_CVw08PX.csv')

    model_text.fit(X_train, y_train, epochs=10, batch_size=128)
    #model_images.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
    model_text.save('model_text_train.h5')
    #model_images.save('model_images_train.h5')
    #y_pred=model_text.predict(X_test)
    #print(y_pred)
    


if __name__ == "__main__":
    main()
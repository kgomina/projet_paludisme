import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Chargement du modèle
model = load_model("mon_modele.h5")

# Interface
st.title("Classification d'image avec Deep Learning")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))  # adapter selon le modèle
    st.image(img, caption='Image chargée', use_column_width=True)

    # Prétraitement
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prédiction
    prediction = model.predict(img_array)
    st.write("Prédiction :", prediction)

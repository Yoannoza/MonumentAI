import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

# Charger le modèle YOLO
model = YOLO("yolov8n.pt")  # charger un modèle officiel

st.title("Détection d/'objets avec YOLOv8")

# Uploader une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    
    # Afficher l'image téléchargée
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("")
    st.write("Détection en cours...")
    
    # Sauvegarder l'image téléchargée pour que YOLO puisse la lire
    image.save("uploaded_image.jpg")
    
    # Prédiction avec le modèle
    results = model("uploaded_image.jpg")
    
    # Sauvegarder le résultat
    results[0].save("result.jpg")
    
    # Afficher le résultat
    result_image = Image.open("result.jpg")
    st.image(result_image, caption='Résultat de la détection', use_column_width=True)

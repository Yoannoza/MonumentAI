# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import math

# # Démarrer la webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # Largeur de la fenêtre de capture
# cap.set(4, 480)  # Hauteur de la fenêtre de capture

# # Modèle
# model = YOLO("yolo-Weights/yolov8n.pt")

# # Classes d'objets
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]

# st.title("Détection d'objets avec YOLO et Streamlit")

# # Démarrer la capture vidéo
# run = st.checkbox('Lancer la détection')

# # Afficher le flux vidéo
# FRAME_WINDOW = st.image([])

# while run:
#     success, img = cap.read()

#     # Vérifier si l'image a été capturée correctement
#     if not success:
#         st.error("Erreur de capture d'image")
#         continue

#     results = model(img, stream=True)

#     # Coordonnées
#     for r in results:
#         boxes = r.boxes

#         for box in boxes:
#             # Boîte englobante
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir en valeurs entières

#             # Placer la boîte dans l'image
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # Confiance
#             confidence = math.ceil((box.conf[0] * 100)) / 100
#             print("Confidence --->", confidence)

#             # Nom de la classe
#             cls = int(box.cls[0])
#             print("Class name -->", classNames[cls])

#             # Détails de l'objet
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2

#             cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

#     # Convertir l'image pour l'affichage dans Streamlit
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(img)

# cap.release()


import threading

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer

lock = threading.Lock()
img_container = {"img": None}


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

fig_place = st.empty()
fig, ax = plt.subplots(1, 1)

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)
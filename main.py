from ultralytics import YOLO 
import cv2
import math 

# Démarrer la webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Largeur de la fenêtre de capture
cap.set(4, 480)  # Hauteur de la fenêtre de capture

# Modèle
model = YOLO("yolo-Weights/yolov8n.pt")

# Classes d'objets
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()

    # Vérifier si l'image a été capturée correctement
    if not success:
        print("Erreur de capture d'image")
        continue

    results = model(img, stream=True)

    # Coordonnées
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Boîte englobante
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir en valeurs entières

            # Placer la boîte dans l'image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confiance
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Nom de la classe
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Détails de l'objet
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

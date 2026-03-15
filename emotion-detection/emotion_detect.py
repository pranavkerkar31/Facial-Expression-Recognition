import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -----------------------------
# Emotion Labels (RAF-DB order)
# -----------------------------
emotion_labels = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

# -----------------------------
# Load ResNet50 Model
# -----------------------------
model = models.resnet50(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 7)
)

model.load_state_dict(torch.load("emotion_resnet50.pth", map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Face Detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80,80)
    )

    for (x,y,w,h) in faces:

        # Expand crop slightly
        pad = int(0.2 * w)

        x1 = max(0, x-pad)
        y1 = max(0, y-pad)
        x2 = min(frame.shape[1], x+w+pad)
        y2 = min(frame.shape[0], y+h+pad)

        face = frame[y1:y2, x1:x2]

        # Draw rectangle
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Convert BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Preprocess
        face = transform(face).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():

            outputs = model(face)

            probs = F.softmax(outputs, dim=1)

            emotion_index = torch.argmax(probs).item()

            confidence = probs[0][emotion_index].item()

        emotion = emotion_labels[emotion_index]

        text = f"{emotion} {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained ResNet model
model = load_model("emotion_resnet_model.keras", compile=False)

# Emotion labels (FER2013)
emotion_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize for model
        face = cv2.resize(face,(224,224))

        # Normalize
        face = face / 255.0

        # Reshape for CNN
        face = np.reshape(face,(1,224,224,3))

        # Predict emotion
        preds = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        # Show emotion text
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
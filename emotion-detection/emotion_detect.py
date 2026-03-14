import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_resnet_model.keras", compile=False)

# Emotion labels (FER2013 order)
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

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(80,80)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize exactly like training
        face = cv2.resize(face,(224,224))

        # Convert BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize
        face = face / 255.0

        # Reshape for model
        face = np.reshape(face,(1,224,224,3))

        # Predict emotion
        preds = model.predict(face, verbose=0)

        emotion_index = np.argmax(preds)
        emotion = emotion_labels[emotion_index]

        # Display emotion
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    # Show webcam window
    cv2.imshow("Emotion Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("emotion_resnet_model.keras", compile=False)
print("Model loaded successfully")
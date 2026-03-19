# Facial Expression Recognition using Deep Learning

## **Project Overview**

This project focuses on **detecting human emotions from facial expressions using deep learning**.  
The system analyzes facial images and classifies them into different emotional categories.

The model detects the following emotions:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

The trained model is integrated with **OpenCV** for **real-time emotion detection using a webcam**.

---

# Dataset

The model was trained using the **FER2013 dataset**, which contains facial expression images labeled with emotions.

## **Dataset Details**

- **Total Images:** ~35,000  
- **Image Size:** **48 × 48 pixels**  
- **Image Type:** **Grayscale**  
- **Classes:** **7 emotions**

The dataset is divided into:

- **Training set**
- **Testing set**

---

# Data Preprocessing

Before training the model, several preprocessing steps were performed.

## **Steps**

- Convert images to **grayscale**
- Resize images to **48 × 48**
- Convert images into **NumPy arrays**
- **Normalize pixel values (0–255 → 0–1)**
- Apply **data augmentation**

## **Data Augmentation Techniques**

- **Rotation**
- **Zoom**
- **Horizontal flipping**
- **Width shifting**
- **Height shifting**

These techniques help the model generalize better.

---

# Model Architecture

A **Convolutional Neural Network (CNN)** was used for emotion classification.

## **Network Structure**

### **Input Layer**

- Image size: **48 × 48 × 1**

### **Convolutional Layers**

- Multiple **Conv2D layers**
- **ReLU activation**
- **MaxPooling layers**

### **Regularization**

- **Dropout layers** to prevent overfitting

### **Fully Connected Layers**

- Dense layers to learn complex emotion patterns

### **Output Layer**

- **Softmax activation**
- **7 emotion classes**

---

# Training Strategy

To improve model performance, several techniques were used.

## **Techniques Used**

- **Dropout** to reduce overfitting
- **Class weights** to handle dataset imbalance
- **Early stopping** to stop training automatically
- **Data augmentation** for better generalization

---

# Model Evaluation

The model was evaluated using multiple metrics.

## **Evaluation Metrics**

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve**
- **AUC Score**

## **Final Performance**

| Metric | Score |
|------|------|
| **Accuracy** | **61.56%** |
| **Precision** | **59.83%** |
| **Recall** | **57.47%** |
| **F1 Score** | **58.18%** |

Considering the complexity of the **FER2013 dataset**, this performance is consistent with baseline models.

---

# Real-Time Emotion Detection

After training the model, it was integrated with **OpenCV** for **live emotion recognition**.

## **Workflow**

1. Capture video from **webcam**
2. Detect faces using **Haar Cascade**
3. Extract face region
4. Preprocess the image
5. Predict emotion using the trained model
6. Display predicted emotion on screen

---

# Technologies Used

## **Programming Language**

- **Python**

## **Libraries**

- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

# Project Structure
# Facial Expression Recognition using Deep Learning

## **Project Overview**

This project focuses on **detecting human emotions from facial expressions using deep learning**.  
The system analyzes facial images and classifies them into different emotional categories.

The model detects the following emotions:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

The trained model is integrated with **OpenCV** for **real-time emotion detection using a webcam**.

---

# Dataset

The model was trained using the **FER2013 dataset**, which contains facial expression images labeled with emotions.

## **Dataset Details**

- **Total Images:** ~35,000  
- **Image Size:** **48 × 48 pixels**  
- **Image Type:** **Grayscale**  
- **Classes:** **7 emotions**

The dataset is divided into:

- **Training set**
- **Testing set**

---

# Data Preprocessing

Before training the model, several preprocessing steps were performed.

## **Steps**

- Convert images to **grayscale**
- Resize images to **48 × 48**
- Convert images into **NumPy arrays**
- **Normalize pixel values (0–255 → 0–1)**
- Apply **data augmentation**

## **Data Augmentation Techniques**

- **Rotation**
- **Zoom**
- **Horizontal flipping**
- **Width shifting**
- **Height shifting**

These techniques help the model generalize better.

---

# Model Architecture

A **Convolutional Neural Network (CNN)** was used for emotion classification.

## **Network Structure**

### **Input Layer**

- Image size: **48 × 48 × 1**

### **Convolutional Layers**

- Multiple **Conv2D layers**
- **ReLU activation**
- **MaxPooling layers**

### **Regularization**

- **Dropout layers** to prevent overfitting

### **Fully Connected Layers**

- Dense layers to learn complex emotion patterns

### **Output Layer**

- **Softmax activation**
- **7 emotion classes**

---

# Training Strategy

To improve model performance, several techniques were used.

## **Techniques Used**

- **Dropout** to reduce overfitting
- **Class weights** to handle dataset imbalance
- **Early stopping** to stop training automatically
- **Data augmentation** for better generalization

---

# Model Evaluation

The model was evaluated using multiple metrics.

## **Evaluation Metrics**

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve**
- **AUC Score**

## **Final Performance**

| Metric | Score |
|------|------|
| **Accuracy** | **61.56%** |
| **Precision** | **59.83%** |
| **Recall** | **57.47%** |
| **F1 Score** | **58.18%** |

Considering the complexity of the **FER2013 dataset**, this performance is consistent with baseline models.

---

# Real-Time Emotion Detection

After training the model, it was integrated with **OpenCV** for **live emotion recognition**.

## **Workflow**

1. Capture video from **webcam**
2. Detect faces using **Haar Cascade**
3. Extract face region
4. Preprocess the image
5. Predict emotion using the trained model
6. Display predicted emotion on screen

---

# Technologies Used

## **Programming Language**

- **Python**

## **Libraries**

- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

# Project Structure
Facial-Expression-Recognition
│
├── dataset
│ ├── train
│ └── test
│
├── emotion_detection.py
├── model_training.ipynb
├── requirements.txt
└── README.md


---

# Future Improvements

Possible improvements include:

- Training with **larger datasets** such as **RAF-DB or AffectNet**
- Using **advanced architectures like EfficientNet**
- Improving **real-time prediction stability**
- Deploying the model as a **web application**

---

# Conclusion

This project demonstrates how **deep learning and computer vision** can be used to recognize human emotions from facial expressions.

The system achieves **reasonable accuracy and real-time performance**, making it useful for applications such as **human-computer interaction, sentiment analysis, and behavioral studies**.

## Download Trained Model

The trained model file is large (~200MB) and cannot be uploaded directly to GitHub.

Download it from Google Drive:

https://drive.google.com/drive/folders/140AL4hnIQh7zTbw9V39OYf9f0nCgLC9B?usp=drive_link

After downloading, place the model file in the project folder:

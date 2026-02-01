# 🎙️ Emotion Recognition from Speech using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

Human emotions play a crucial role in communication. This project focuses on **automatic emotion recognition from speech audio** using **speech signal processing** and **deep learning** techniques.

The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** from speech signals and uses a **Convolutional Neural Network (CNN)** to classify emotions.

This project is developed as part of **CodeAlpha Internship – Task 2**.

---

## 🎯 Objectives

- Extract meaningful features from speech signals  
- Train a deep learning model to classify emotions  
- Evaluate model accuracy on unseen data  
- Predict emotions from new speech audio  

---

## 🧠 Emotions Classified

The model recognizes the following **8 emotions**:

- 😐 Neutral  
- 😌 Calm  
- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😨 Fear  
- 🤢 Disgust  
- 😲 Surprise  

---

## 🧪 Dataset Used

### 🎼 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- 1440+ labeled speech samples  
- 24 professional actors  
- High-quality `.wav` audio files  
- Emotion labels encoded in filenames  

### Why RAVDESS?
- Widely used academic dataset  
- Balanced emotion classes  
- Clean and standardized recordings  

---

## 📁 Project Structure

## 📁 Project Structure

```text
CodeAlpha_Emotion_Recognition_from_Speech/
│
├── RAVDESS
│
├── models/
│   └── emotion_model.h5
│
├── src/
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore

```
---
## ⚙️ Technologies Used

### 🔹 Programming Language
- Python 3.9+

### 🔹 Libraries & Frameworks
- TensorFlow / Keras – Deep Learning  
- Librosa – Speech & audio processing  
- NumPy – Numerical computations  
- Scikit-learn – Label encoding & evaluation  
- Matplotlib / Seaborn – Visualization  

---

## 🔍 Feature Extraction

### 🎵 MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs are used because:
- They closely represent human auditory perception  
- Effective for speech and emotion recognition  
- Reduce noise and irrelevant information  

Each audio file is converted into **40 MFCC features**.

---

## 🧠 Model Architecture

### 📌 Convolutional Neural Network (CNN)

**Why CNN?**
- Learns spatial patterns from MFCC features  
- Faster training with fewer parameters  
- Strong performance on audio-based tasks  

---

## 📊 Model Performance

- Train/Test Split: **80% / 20%**
- Evaluation Metric: **Accuracy**
- Achieved Accuracy: **~70–75%**

> Accuracy may vary slightly due to random initialization and data splitting.

---
## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Shubhra-kanti/CodeAlpha_Emotion_Recognition_from_Speech.git
cd CodeAlpha_Emotion_Recognition_from_Speech
```
2️⃣ Create and Activate Virtual Environment
```bash
python -m venv myenv
myenv\Scripts\activate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Train the Model
```bash
python src/train_model.py
```
5️⃣ Test Emotion on New Audio
```bash
python src/test_audio.py
```
🧪 Sample Output
Predicted Emotion: Happy 😊

🔮 Future Enhancements

Data augmentation for improved accuracy

LSTM / BiLSTM models for temporal learning

Web interface using Flask or Streamlit

Real-time emotion recognition

👨‍💻 Author

Shubhra Kanti Banerjee
Engineering Student
West Bengal, India

📜 License

This project is intended for educational and research purposes only.

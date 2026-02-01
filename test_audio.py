import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

model = load_model("emotion_model.h5")

emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

file_path = "anger-scream.wav"   # put test audio here
features = extract_mfcc(file_path)
features = features.reshape(1, 40, 1)

prediction = model.predict(features)
print("Predicted Emotion:", emotion_labels[np.argmax(prediction)])

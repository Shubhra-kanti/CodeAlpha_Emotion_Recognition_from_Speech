import librosa
import numpy as np
import os

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def load_dataset(dataset_path):
    X, y = [], []

    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise"
    }

    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            if not file.endswith(".wav"):
                continue

            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)

            if emotion is None:
                continue

            file_path = os.path.join(actor_path, file)
            features = extract_mfcc(file_path)

            X.append(features)
            y.append(emotion)

    return np.array(X), np.array(y)


import numpy as np
from extract_features import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# Load data
X, y = load_dataset(r"D:\SHUBHRA KANTI BANERJEE\Internship\Emotion Recognition from Speech\RAVDESS")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1)

# Build model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trainin model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("D:\SHUBHRA KANTI BANERJEE\Internship\Emotion Recognition from Speech\emotion_model.h5")

print("Model trained and saved successfully.")

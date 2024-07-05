import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def load_data():
    gestures = ["Open_Hand", "Closed_Fist", "Pointing", "Peace_Sign", "Thumbs_Up", "OK_Sign"]
    X = []
    y = []

    for i, gesture in enumerate(gestures):
        path = f"dataset/{gesture}"
        for file in os.listdir(path):
            landmarks = np.load(os.path.join(path, file))
            X.append(landmarks)
            y.append(i)

    return np.array(X), np.array(y)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

model.save('gesture_model.h5')
print("Model saved as 'gesture_model.h5'")

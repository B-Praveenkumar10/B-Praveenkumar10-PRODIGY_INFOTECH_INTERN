import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

model = load_model('gesture_model.h5')
gestures = ["Open_Hand", "Closed_Fist", "Pointing", "Peace_Sign", "Thumbs_Up", "OK_Sign"]

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).flatten()
            prediction = model.predict(landmarks.reshape(1, -1))
            gesture_index = np.argmax(prediction)
            gesture = gestures[gesture_index]

            cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
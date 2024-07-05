import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    dataset_size = 100
    gestures = ["Open_Hand", "Closed_Fist", "Pointing", "Peace_Sign", "Thumbs_Up", "OK_Sign"]

    for gesture in gestures:
        if not os.path.exists(f"dataset/{gesture}"):
            os.makedirs(f"dataset/{gesture}")

        print(f"Collecting data for {gesture}. Press 's' to start and 'q' to quit.")
        count = 0
        collecting = False

        while count < dataset_size:
            success, image = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting:
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).flatten()
                    np.save(f"dataset/{gesture}/{count}.npy", landmarks)
                    count += 1

            cv2.putText(image, f"{gesture}: {count}/{dataset_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Press 's' to start/pause, 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = not collecting
                print("Collection " + ("started" if collecting else "paused"))
            elif key == ord('q'):
                break

        if count < dataset_size:
            print(f"Warning: Only collected {count} samples for {gesture}")

    cap.release()
    cv2.destroyAllWindows()

collect_data()
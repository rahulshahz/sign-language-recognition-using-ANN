# DEPENDENCIES ------>
import cv2
import mediapipe as mp
import numpy as np
import sys
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("model.h5")

# Initialize the webcam
capture = cv2.VideoCapture(0)

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Sign language labels dictionary
labels_dict = {0: "hello", 1: "i love you", 2: "yes", 3: "good", 4: "bad", 5: "okay", 6: "you", 7: "I am", 8: "why", 9: "no", 10: 'A'}
reversed_labels_dict = {v: k for k, v in labels_dict.items()}
no_of_labels = 11
d = {}
for i in range(no_of_labels):
    d[i] = (0, 0)

count = 0
res = []

while True:
    count += 1
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = capture.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Ensure data_aux has the correct shape
            # data_aux.extend([0] * (84 - len(data_aux)))  # Pad to 84 values
            data_aux = np.array(data_aux).reshape(1, 42)  # Reshape to (1, 84)
            prediction = model.predict(data_aux)  # Use the reshaped data_aux
            
            check = np.argmax(prediction[0])
            print(check)
            sign_name = labels_dict[check]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, sign_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            data_aux = []
            x_ = []
            y_ = []

    cv2.imshow("Handy", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
sys.exit(0)

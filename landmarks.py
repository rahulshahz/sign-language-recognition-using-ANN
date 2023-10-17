import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define the directory where image data is stored
data_dir = "./data"

# Initialize lists to store data and corresponding labels
data = []
labels = []

# Iterate through each subdirectory in the data directory
for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = []  # Initialize a list to store hand landmark data for an image
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))  # Read an image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

        # Use MediaPipe Hands to process the image and detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # X-coordinate of a hand landmark
                    y = hand_landmarks.landmark[i].y  # Y-coordinate of a hand landmark
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)  # Append the hand landmark data to the data list
            labels.append(dir_)  # Append the corresponding label to the labels list

# Create a binary file to store the data and labels using Pickle
file = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, file)
file.close()

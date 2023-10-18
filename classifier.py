# DEPENDENCIES ------>
import cv2
import mediapipe as mp
import numpy as np
import sys
from tensorflow import keras

# Load the trained model from a .h5 file
model = keras.models.load_model("model.h5")

# Initialize the webcam for capturing video
capture = cv2.VideoCapture(0)

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Define a dictionary that maps numerical labels to sign language words
labels_dict = {0: "A", 1: "I am", 2: "bad", 3: "good", 4: "hello", 5: "I love you", 6: "no", 7: "okay", 8: "why", 9: "yes",10:"you"}

# Create a reversed dictionary to map sign language words back to numerical labels
reversed_labels_dict = {v: k for k, v in labels_dict.items()}

# Define the number of labels
no_of_labels = 11

# Initialize a dictionary for tracking counts
d = {}
for i in range(no_of_labels):
    d[i] = (0, 0)

# Initialize counters and storage variables
count = 0
res = []

# Start an infinite loop to capture video from the webcam
while True:
    count += 1
    data_aux = []  # Initialize a list to store hand landmark data
    x_ = []  # Lists to store X coordinates of landmarks
    y_ = []  # Lists to store Y coordinates of landmarks

    ret, frame = capture.read()  # Read a frame from the webcam
    H, W, _ = frame.shape  # Get the height (H) and width (W) of the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format

    results = hands.process(frame_rgb)  # Use MediaPipe Hands to process the frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Loop through all landmarks in the hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # X-coordinate of a hand landmark
                y = hand_landmarks.landmark[i].y  # Y-coordinate of a hand landmark
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10  # Calculate the coordinates of a bounding box
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Ensure data_aux has the correct shape
            data_aux.extend(data_aux)
            data_aux = np.array(data_aux).reshape(1, 84)  # Reshape to (1, 42)
            
            # Make a prediction using the loaded model
            prediction = model.predict(data_aux)
            check = np.argmax(prediction[0])  # Find the index of the maximum prediction
            sign_name = labels_dict[check]  # Get the corresponding sign language word
            
            # Draw a bounding box and the predicted sign language word on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, sign_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            data_aux = []  # Clear the data for the next frame
            x_ = []
            y_ = []

    cv2.imshow("Handy", frame)  # Display the frame with drawn landmarks and predictions
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press

    if key == ord('q'):  # If the 'q' key is pressed, exit the loop
        break

# Release the webcam, close OpenCV windows, and exit the program
capture.release()
cv2.destroyAllWindows()
sys.exit(0)

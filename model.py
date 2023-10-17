# DEPENDENCIES ------>
import os
import cv2

# Define the directory where data will be saved
DATA_DIR = "./data"

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the dataset size for each class
num_classes = 10
dataset_size = 1000

# Define a dictionary to map class labels to class names
labels_dict = {0: "hello", 1: "i love you", 2: "yes"}

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Iterate through each class label
for key, value in labels_dict.items():
    # Create a directory for the current class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, value)):
        os.makedirs(os.path.join(DATA_DIR, value))
    print("Collecting: " + value)

    # Wait for the user to press '1' to start capturing images for the current class
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press '1' ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        
        # If the user presses '1', exit the loop to start capturing images
        if cv2.waitKey(25) == ord("1"):
            break

    count = 0
    # Capture and save the specified number of images for the current class
    while count < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        cv2.waitKey(25)
        
        # Save each captured frame as an image in the corresponding class directory
        cv2.imwrite(os.path.join(DATA_DIR, value, "{}.jpg".format(count)), frame)
        count += 1

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

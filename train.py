import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras

# Load your data
data_dict = pickle.load(open("./data1.pickle", "rb"))
data = data_dict["data"]
labels = data_dict["labels"]

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Find the maximum sequence length
max_length = max(len(seq) for seq in data)

# Pad the sequences to the maximum length
data_padded = np.array([np.pad(seq, (0, max_length - len(seq))) for seq in data])

# Shuffle the data and labels
data_padded, labels_encoded = shuffle(data_padded, labels_encoded)

# Create an improved neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(max_length,)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Larger first hidden layer
    keras.layers.Dropout(0.5),  # Dropout layer for regularization
    keras.layers.Dense(256, activation='relu'),  # A second hidden layer
    keras.layers.Dropout(0.5),  # Dropout layer for regularization
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Adjust learning rate
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the number of folds for cross-validation
n_splits = 5

# Initialize StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create a list to store accuracy for each fold
accuracies = []

# Perform cross-validation
for train_index, test_index in cv.split(data_padded, labels_encoded):
    X_train, X_test = data_padded[train_index], data_padded[test_index]
    y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]
    
    # Train the model on the training data
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
    
    # Evaluate the model on the test data
    accuracy = model.evaluate(X_test, y_test, verbose=1)[1]
    accuracies.append(accuracy)

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")

# Save the model
model.save("model.h5")

import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras

# Load your data
data_dict = pickle.load(open("./data.pickle", "rb"))
data = data_dict["data"]
labels = data_dict["labels"]

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Find the maximum sequence length
max_length = max(len(seq) for seq in data)

# Pad the sequences to the maximum length
data_padded = np.array([np.pad(seq, (0, max_length - len(seq))) for seq in data])

# Shuffle the data and labels for cross-validation
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

# Train the model (you may need to adjust the number of epochs and batch size)
model.fit(data_padded, labels_encoded, epochs=50, batch_size=64, validation_split=0.2, verbose=1)


# Calculate the cross-validation score using StratifiedKFold
cv = StratifiedKFold(n_splits=5)
accuracy = model.evaluate(data_padded, labels_encoded, verbose=0)
print(f"Accuracy: {accuracy[1] * 100:.2f}%")
model.save("model.h5")

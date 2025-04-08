import os
import time
from time import sleep

import numpy as np
import pandas as pd
from ctypes import *

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image


class MLP(Structure):
    _fields_ = [
        ("weight_input_hidden", POINTER(c_double)),
        ("weight_output_hidden", POINTER(c_double)),
        ("bias_hidden", POINTER(c_double)),
        ("bias_output", POINTER(c_double)),
        ("learning_rate", c_double),
        ("hidden_input", POINTER(c_double)),
        ("hidden_output", POINTER(c_double)),
        ("final_input", POINTER(c_double)),
        ("final_output", POINTER(c_double)),
        ("input_size", c_size_t),
        ("hidden_size", c_size_t),
        ("output_size", c_size_t),
    ]


# Load the Rust library
lib = CDLL('./dll/yetnewmlp.dll')

lib.mlp_new.restype = POINTER(MLP)
lib.mlp_new.argtypes = [c_size_t, c_size_t, c_size_t, c_double]
lib.mlp_forward.restype = POINTER(c_double)
lib.mlp_forward.argtypes = [POINTER(MLP), POINTER(c_double), c_size_t, c_size_t]
lib.mlp_train.restype = None
lib.mlp_train.argtypes = [POINTER(MLP), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t, c_size_t,
                          c_size_t, c_size_t]
lib.mlp_free.restype = None
lib.mlp_free.argtypes = [POINTER(MLP)]


def numpy_to_ctypes(array):
    return array.ctypes.data_as(POINTER(c_double))


def ctypes_to_numpy(ptr, shape):
    size = np.prod(shape)
    return np.ctypeslib.as_array(ptr, shape=(size,)).reshape(shape)


# Load dataset
data = pd.read_csv('another_image_dataset.csv')

# Split dataset into features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to one-hot encoding
y_train_onehot = np.eye(3)[y_train]
y_test_onehot = np.eye(3)[y_test]

# Parameters
input_size = X_train.shape[1]
hidden_size = 512  # You can adjust this value
output_size = 3
learning_rate = 0.001  # 0.00007
epochs = 40
batch_size = 32

# Create an MLP
mlp = lib.mlp_new(input_size, hidden_size, output_size, c_double(learning_rate))

# Prepare input data
X_train_ptr = numpy_to_ctypes(X_train)
y_train_ptr = numpy_to_ctypes(y_train_onehot)

begin = time.time()
# Train the MLP
lib.mlp_train(mlp, X_train_ptr, y_train_ptr, X_train.shape[0], X_train.shape[1], y_train_onehot.shape[0],
              y_train_onehot.shape[1], epochs, batch_size)
print("Duration to train ", time.time() - begin)
# Forward pass for predictions
X_test_ptr = numpy_to_ctypes(X_test)
output_ptr = lib.mlp_forward(mlp, X_test_ptr, X_test.shape[0], X_test.shape[1])
output = ctypes_to_numpy(output_ptr, (X_test.shape[0], output_size))

# Get predicted labels
y_pred = np.argmax(output, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


def predict_single(img_path, target_size=(28, 28), padding_color=255):
    image = Image.open(img_path).convert('L').resize(target_size)

    image_array = np.array(image).astype(np.float64)
    image_array = image_array / 255.0

    # Flatten the image array
    image_array = image_array.flatten()

    single_test_ptr = numpy_to_ctypes(image_array)

    single_output_ptr = lib.mlp_forward(mlp, single_test_ptr, 1, image_array.shape[0])
    single_output = ctypes_to_numpy(single_output_ptr, (1, output_size))

    y_pred = np.argmax(single_output, axis=1)[0]
    classes = ["plane", "car", "bike"]

    print("Predicted:", classes[y_pred])


image_dir = './DATA/test/'
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        print(f"Predicting image: {image_path}")
        predict_single(image_path)
        sleep(1)

# Test x_test, y_test
total = len(X_test)
correct = 0
for i in range(len(X_test)):
    # Prepare input for the forward pass
    test = X_test[i]
    expect = y_test[i]
    single_test_ptr = numpy_to_ctypes(test)

    # Perform forward pass
    single_output_ptr = lib.mlp_forward(mlp, single_test_ptr, 1, test.shape[0])
    single_output = ctypes_to_numpy(single_output_ptr, (1, output_size))

    # Get predicted labels
    y_pred = np.argmax(single_output, axis=1)[0]
    classes = ["plane", "car", "bike"]
    if y_pred == expect:
        correct += 1
    print("Predicted:", classes[y_pred], classes[expect])

print(f"Test accuracy: {correct / total *100}%")


# Free the MLP
lib.mlp_free(mlp)

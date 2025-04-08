import os
from time import sleep

import numpy as np
import pandas as pd
from ctypes import *
from tensorflow.keras.datasets import mnist

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


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

input_size = X_train.shape[1]
hidden_size = 512
output_size = 3
learning_rate = 0.01
epochs = 20
batch_size = 32

mlp = lib.mlp_new(input_size, hidden_size, output_size, c_double(learning_rate))

X_train_ptr = numpy_to_ctypes(X_train)
y_train_ptr = numpy_to_ctypes(y_train_onehot)

lib.mlp_train(mlp, X_train_ptr, y_train_ptr, X_train.shape[0], X_train.shape[1], y_train_onehot.shape[0],
              y_train_onehot.shape[1], epochs, batch_size)

X_test_ptr = numpy_to_ctypes(X_test)
output_ptr = lib.mlp_forward(mlp, X_test_ptr, X_test.shape[0], X_test.shape[1])
output = ctypes_to_numpy(output_ptr, (X_test.shape[0], output_size))

y_pred = np.argmax(output, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

def predict_single(img, target_size=(28, 28), padding_color=255):
    image = Image.open(img).convert('L')

    resized_image = image.resize(target_size)

    image_array = np.asarray(resized_image).astype(np.float64) / 255.0

    image_array = image_array.flatten()

    single_test_ptr = numpy_to_ctypes(image_array)

    single_output_ptr = lib.mlp_forward(mlp, single_test_ptr, 1, image_array.shape[0])
    single_output = ctypes_to_numpy(single_output_ptr, (1, output_size))

    y_pred = np.argmax(single_output, axis=1)[0]
    classes = [str(i) for i in range(10)]

    print(f"Predicted:{y_pred}" )



image_dir = '.\\DATA\\testabcxyz'
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        print(image_path)
        predict_single(image_path)

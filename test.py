import os
import time
from io import BytesIO
from time import sleep
import base64
from io import BytesIO
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.datasets import mnist


# Define the MLP class and functions as in the previous code

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)

        # Update weights and biases for hidden-to-output layer
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= self.learning_rate * np.sum(output_error, axis=0, keepdims=True)

        # Update weights and biases for input-to-hidden layer
        self.weights_input_hidden -= self.learning_rate * np.dot(x.T, hidden_delta)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

        l2_reg = 0.01  # Regularization strength
        self.weights_hidden_output -= l2_reg * self.weights_hidden_output
        self.weights_input_hidden -= l2_reg * self.weights_input_hidden

    def train(self, x, y, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                output = self.forward(batch_x)
                self.backward(batch_x, batch_y, output)
                total_loss += -np.sum(batch_y * np.log(output + 1e-8))
            print(f"Epoch {epoch}, Loss: {total_loss / len(x)}")

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((28, 28))  # Resize to 28x28
#
#     # Invert the image if the background is white
#     image_array = np.asarray(image)
#     if np.mean(image_array) > 127:  # If the mean pixel value is greater than 127, invert the image
#         image_array = 255 - image_array
#
#     # Normalize the image
#     image_array = image_array.astype(np.float64) / 255.0
#     image_array = image_array.flatten()  # Flatten the image to a 1D array of shape (784,)
#     return image_array
def preprocess_image(image_data):
    # Decode the base64 string
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28

    # Invert the image if the background is white
    image_array = np.asarray(image)
    if np.mean(image_array) > 127:  # If the mean pixel value is greater than 127, invert the image
        image_array = 255 - image_array

    # Normalize the image
    image_array = image_array.astype(np.float64) / 255.0
    image_array = image_array.flatten()  # Flatten the image to a 1D array of shape (784,)
    return image_array


# Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train[:30000]
# y_train = y_train[:30000]
# # Preprocess the MNIST data
# X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
# X_test = x_test.reshape(x_test.shape[0], -1) / 255.0
# y_train_one_hot = one_hot_encode(y_train)
#
# # Initialize the MLP
# input_size = 784  # 28x28 pixels
# hidden_size = 512  # You can change this value
# output_size = 10  # Number of classes
# learning_rate = 0.009  # Adjusted learning rate
# epochs = 10


def load_and_train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, y_train = x_train[:30000], y_train[:30000]  # Use a subset for faster training

    X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    X_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train_one_hot = one_hot_encode(y_train)

    input_size = 784
    hidden_size = 512
    output_size = 10
    learning_rate = 0.01
    epochs = 20

    mlp = MLP(input_size, hidden_size, output_size, learning_rate)
    start_time = time.time()

    mlp.train(X_train, y_train_one_hot, epochs, batch_size=50)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return mlp


# load_and_train_model()

# mlp = MLP(input_size, hidden_size, output_size, learning_rate)
#
# # Train the MLP
# mlp.train(X_train, y_train_one_hot, epochs, batch_size=50)
#
# correct = 0
# image_dir = 'D:\\Download\\new_PA\\NEW_FACKING_MLP\\DATA\\testabcxyz'
# for filename in os.listdir(image_dir):
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         image_path = os.path.join(image_dir, filename)
#         processed_image = preprocess_image(image_path)
#         processed_image = processed_image.reshape(1, -1)  # Reshape for the model
#         prediction = mlp.predict(processed_image)
#         print(f'Predicted Label for {filename}: {prediction[0]} ')
#
# # Test with the test set
# predictions = mlp.predict(X_test)
# accuracy = np.mean(predictions == y_test)
# print(f"Test Accuracy: {accuracy}")
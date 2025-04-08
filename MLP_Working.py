import os

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from numba import vectorize, jit, prange, cuda
import time


@vectorize(['float32(float32)'], target='cuda')
def relu(x):
    return max(0, x)


@vectorize(['float32(float32)'], target='cuda')
def relu_derivative(x):
    return np.float32(1.0) if x > 0 else np.float32(0.0)


@jit(nopython=True, parallel=True, target_backend='cuda')
def softmax(x):
    result = np.empty_like(x, dtype=np.float32)
    for i in prange(x.shape[0]):
        exps = np.exp(x[i] - np.max(x[i]))
        result[i] = exps / np.sum(exps)
    return result


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases with float32 dtype
        self.weights_input_hidden = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size), dtype=np.float32)
        self.bias_output = np.zeros((1, output_size), dtype=np.float32)
        self.learning_rate = np.float32(learning_rate)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden).astype(np.float32) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output).astype(np.float32) + self.bias_output
        self.final_output = softmax(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu_derivative(self.hidden_output)

        self.weights_hidden_output -= (self.learning_rate * np.dot(self.hidden_output.T, output_error)).astype(
            np.float32)
        self.bias_output -= (self.learning_rate * np.sum(output_error, axis=0, keepdims=True)).astype(np.float32)

        self.weights_input_hidden -= (self.learning_rate * np.dot(x.T, hidden_delta)).astype(np.float32)
        self.bias_hidden -= (self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)).astype(np.float32)

        l2_reg = np.float32(0.01)
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





# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
y_train_one_hot = one_hot_encode(y_train).astype(np.float32)

# Initialize the MLP
input_size = 784  # 28x28 pixels
hidden_size = 512  # You can change this value
output_size = 10  # Number of classes
learning_rate = 0.01 # Adjusted learning rate
epochs = 20

mlp = MLP(input_size, hidden_size, output_size, learning_rate)
start_time = time.time()

# Train the MLP
mlp.train(X_train, y_train_one_hot, epochs, 50)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Evaluate the MLP
predictions = mlp.predict(X_test)
print(predictions)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

correct = 0
image_dir = 'D:\\Download\\new_PA\\NEW_FACKING_MLP\\DATA\\testabcxyz'
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        processed_image = preprocess_image(image_path)
        processed_image = processed_image.reshape(1, -1)  # Reshape for the model
        prediction = mlp.predict(processed_image)
        print(f'Predicted Label for {filename}: {prediction[0]} ')

# Test with the test set
predictions = mlp.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy}")

import os

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from numba import vectorize, jit, prange, cuda
import time

@jit(nopython=True, parallel=True, target_backend='cuda')
def softmax(x):
    result = np.empty_like(x, dtype=np.float32)
    for i in prange(x.shape[0]):
        exps = np.exp(x[i] - np.max(x[i]))
        result[i] = exps / np.sum(exps)
    return result

class LR:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
        self.bias = np.zeros((1, output_size), dtype=np.float32)
        self.learning_rate = np.float32(learning_rate)

    def cal_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, x):
        logits = np.dot(x, self.weights) + self.bias
        probs = softmax(logits)
        return probs

    def backward(self, x, y, probs):
        m = x.shape[0]
        grad_w = np.dot(x.T, (probs - y)) / m
        grad_b = np.sum(probs - y, axis=0, keepdims=True) / m
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
    def train(self, x, y, epochs):
        for epoch in range(epochs):
            probs = self.forward(x)
            loss = self.cal_loss(probs, y)
            self.backward(x, y, probs)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
y_train_one_hot = one_hot_encode(y_train).astype(np.float32)

input_size = 784
output_size = 10
learning_rate = 0.5
epochs = 400

model = LR(input_size, output_size, learning_rate)
model.train(x_train, y_train_one_hot, epochs)

correct = 0
image_dir = '.\\DATA\\testabcxyz'
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        processed_image = preprocess_image(image_path)
        processed_image = processed_image.reshape(1, -1)  # Reshape for the model
        prediction = model.predict(processed_image)
        print(f'Predicted Label for {filename}: {prediction[0]} ')

predictions = model.predict(x_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

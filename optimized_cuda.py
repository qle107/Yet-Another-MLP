import os
import numpy as np
import cupy as cp
from PIL import Image
from tensorflow.keras.datasets import mnist
import time

# Use CuPy for GPU operations
xp = cp


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.weights1 = xp.random.randn(input_size, hidden_size).astype(xp.float32) * 0.01
        self.bias1 = xp.zeros((1, hidden_size), dtype=xp.float32)
        self.weights2 = xp.random.randn(hidden_size, hidden_size).astype(xp.float32) * 0.01
        self.bias2 = xp.zeros((1, hidden_size), dtype=xp.float32)
        self.weights3 = xp.random.randn(hidden_size, output_size).astype(xp.float32) * 0.01
        self.bias3 = xp.zeros((1, output_size), dtype=xp.float32)
        self.learning_rate = learning_rate

    def relu(self, x):
        return xp.maximum(0, x)

    def relu_derivative(self, x):
        return xp.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
        return exp_x / xp.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.layer1 = self.relu(xp.dot(x, self.weights1) + self.bias1)
        self.layer2 = self.relu(xp.dot(self.layer1, self.weights2) + self.bias2)
        self.output = self.softmax(xp.dot(self.layer2, self.weights3) + self.bias3)
        return self.output

    def backward(self, x, y, output):
        output_error = output - y
        d_layer2 = xp.dot(output_error, self.weights3.T) * self.relu_derivative(self.layer2)
        d_layer1 = xp.dot(d_layer2, self.weights2.T) * self.relu_derivative(self.layer1)

        self.weights3 -= self.learning_rate * xp.dot(self.layer2.T, output_error)
        self.bias3 -= self.learning_rate * xp.sum(output_error, axis=0, keepdims=True)
        self.weights2 -= self.learning_rate * xp.dot(self.layer1.T, d_layer2)
        self.bias2 -= self.learning_rate * xp.sum(d_layer2, axis=0, keepdims=True)
        self.weights1 -= self.learning_rate * xp.dot(x.T, d_layer1)
        self.bias1 -= self.learning_rate * xp.sum(d_layer1, axis=0, keepdims=True)

    def train(self, x, y, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                output = self.forward(batch_x)
                self.backward(batch_x, batch_y, output)
                total_loss += -xp.sum(batch_y * xp.log(output + 1e-8))
            print(f"Epoch {epoch}, Loss: {total_loss / len(x)}")

    def predict(self, x):
        output = self.forward(x)
        return xp.argmax(output, axis=1)


def one_hot_encode(labels, num_classes=10):
    return xp.eye(num_classes)[labels]


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = xp.asarray(image)
    if xp.mean(image_array) > 127:
        image_array = 255 - image_array
    image_array = image_array.astype(xp.float32) / 255.0
    return image_array.flatten()


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
y_train_one_hot = one_hot_encode(y_train).astype(np.float32)

# Initialize the MLP
input_size = 784  # 28x28 pixels
hidden_size = 512  # You can change this value
output_size = 10  # Number of classes
learning_rate = 0.0015  # Adjusted learning rate
epochs = 2000

mlp = MLP(input_size, hidden_size, output_size, learning_rate)
start_time = time.time()

# Train the MLP
mlp.train(X_train, y_train_one_hot, epochs, 256)
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

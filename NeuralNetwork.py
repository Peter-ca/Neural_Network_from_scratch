import numpy as np
import pandas as pd
from PIL import Image
import pickle

# Load MNIST dataset
data = pd.read_csv(r'mnist_train.csv')
data = np.array(data)
m, _ = data.shape
np.random.shuffle(data)
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:]
X_train = X_train / 255.

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.rand(10, 784)
        self.b1 = np.random.rand(10, 1)
        self.W2 = np.random.rand(10, 10)
        self.b2 = np.random.rand(10, 1)

    def ReLU(self, z):
        return np.maximum(z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A

    def feedforward(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.Z1, self.A1, self.Z2, self.A2

    def one_hot_encoding(self, Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def ReLU_derivative(self, Z):
        return Z > 0

    def backpropagation(self, X, Y):
        one_hot_Y = self.one_hot_encoding(Y)
        dZ2 = self.A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_derivative(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        self.dW1, self.db1 = dW1, db1
        self.dW2, self.db2 = dW2, db2

    def update_parameters(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y):
        learning_rate = 1
        epochs = 100
        for i in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, Y)
            self.update_parameters(learning_rate)
            if i % 10 == 0:
                print("epoch:", i)
                predictions = np.argmax(self.A2, axis=0)
                print(self.get_accuracy(predictions, Y))
        return self.W1, self.b1, self.W2, self.b2

    def train_and_save_model(self):
        self.gradient_descent(X_train, Y_train)
        with open('weights_biases_unethical.pkl', 'wb') as file:
            pickle.dump([self.W1, self.b1, self.W2, self.b2], file)
        return self.W1, self.b1, self.W2, self.b2

    def load_trained_model(self):
        with open('weights_biases.pkl', 'rb') as file:
            self.W1, self.b1, self.W2, self.b2 = pickle.load(file)
        return self.W1, self.b1, self.W2, self.b2

    def test_self_drawn_img(self):
        image = Image.open('number.png').convert('L')
        image = image.resize((28, 28))
        image = np.array(image).reshape(784, 1) / 255
        image = - (image - 1)
        self.load_trained_model()
        _, _, _, A2 = self.feedforward(image)
        prediction = np.argmax(A2, 0)
        return f'Prediction: {prediction[0]}'

if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.train_and_save_model()
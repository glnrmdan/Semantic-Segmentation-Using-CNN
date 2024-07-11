import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.weights_input_hidden1 = self.initialize_weights(input_size, hidden_size1) * 0.01
        self.bias_input_hidden1 = np.zeros((1, hidden_size1))

        self.weights_hidden1_hidden2 = self.initialize_weights(hidden_size1, hidden_size2) * 0.01
        self.bias_hidden1_hidden2 = np.zeros((1, hidden_size2))

        self.weights_hidden2_output = self.initialize_weights(hidden_size2, output_size) * 0.01
        self.bias_hidden2_output = np.zeros((1, output_size))

    def initialize_weights(self, input_size, output_size):
        weights = np.random.randn(input_size, output_size)
        weights = np.clip(weights, -1, 1)  # range bobot -1 -> 1
        return weights

    def forward(self, X):
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_input_hidden1
        self.hidden_output1 = 1 / (1 + np.exp(-self.hidden_input1))

        self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2
        self.hidden_output2 = 1 / (1 + np.exp(-self.hidden_input2))

        self.output = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_hidden2_output
        return self.output 

    def backward(self, X, y, learning_rate):
        output_error = (self.output.T - y.T).T

        hidden2_error = np.dot(output_error, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * self.hidden_output2 * (1 - self.hidden_output2)

        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * self.hidden_output1 * (1 - self.hidden_output1)

        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden_output2.T, output_error)
        self.bias_hidden2_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden_output1.T, hidden2_delta)
        self.bias_hidden1_hidden2 -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)

        self.weights_input_hidden1 -= learning_rate * np.dot(X.T, hidden1_delta)
        self.bias_input_hidden1 -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # mencegah pembagian dengan nol
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # membatasi nilai untuk mencegah pembagian dengan nol
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()  # Calculate binary cross entropy loss
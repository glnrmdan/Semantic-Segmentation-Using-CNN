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
        # H_hidden = X * W_input_hidden1 + B_input_hidden1
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_input_hidden1 # hidden_input1 -> H_hidden
        # A_hidden1 = 1 / (1 + exp(-H_hidden))
        self.hidden_output1 = 1 / (1 + np.exp(-self.hidden_input1)) # -> hidden_output1 -> A_hidden1
        
        # Z_hidden = A_hidden1 * W_hidden1_hidden2 + B_hidden1_hidden2
        self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2 # hidden_input2 -> Z_hidden
        # A_hidden2 = 1 / (1 + exp(-Z_hidden))
        self.hidden_output2 = 1 / (1 + np.exp(-self.hidden_input2)) # hidden_output2 -> A_hidden2
        # Z_output = A_hidden2 * W_hidden2_output + B_hidden2_output
        self.output = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_hidden2_output # Output layer -> Z_output
        return self.output 
    
    def backward(self, X, y, learning_rate):
        # d_output = (y^ - y)
        output_error = (self.output.T - y.T).T # output_error = d_output
        
        # d_hidden2 = d_output * W_hidden2_output * A_hidden2 * (1 - A_hidden2)
        hidden2_error = np.dot(output_error, self.weights_hidden2_output.T) 
        hidden2_delta = hidden2_error * self.hidden_output2 * (1 - self.hidden_output2) # hidden2_delta = d_hidden2
        
        # d_hidden1 = d_hidden2 * W_hidden1_hidden2 * A_hidden1 * (1 - A_hidden1)
        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * self.hidden_output1 * (1 - self.hidden_output1) # hidden1_delta = d_hidden1
        
        # Update bobot dan bias yang menghubungkan hidden layer 2 dan output layer
        # W_hidden2_output = W_hidden2_output - learning_rate * A_hidden2 * d_output
        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden_output2.T, output_error)
        # B_hidden2_output = B_hidden2_output - learning_rate * d_output
        self.bias_hidden2_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        
        # Update bobot dan bias yang menghubungkan hidden layer 1 dan hidden layer 2
        # W_hidden1_hidden2 = W_hidden1_hidden2 - learning_rate * A_hidden1 * d_hidden2
        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden_output1.T, hidden2_delta)
        # B_hidden1_hidden2 = B_hidden1_hidden2 * learning_rate * d_hidden2
        self.bias_hidden1_hidden2 -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        # Update bobot dan bias yang menghubungkan input layer dan hidden layer 1
        # W_input_hidden1 = W_input_hidden1 - learning_rate * X * d_hidden1
        self.weights_input_hidden1 -= learning_rate * np.dot(X.T, hidden1_delta)
        # B_input_hidden1 = B_input_hidden1 - learning_rate * d_hidden1
        self.bias_input_hidden1 -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)
        
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # Small value to prevent division by zero 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip values to prevent division by zero
    # Calculate binary cross entropy (BCE)
    # BCE = -1/N * Σ(y_ij * log(y^_ij) + (1 - y_ij) * log(1 - y^_ij))
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
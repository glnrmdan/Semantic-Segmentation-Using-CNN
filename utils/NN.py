import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = self.initialize_weights(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = self.initialize_weights(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))
        
    # def __init__(self, input_size, hidden_size, output_size):
    #     self.weights_input_hidden = np.random.randn(input_size, hidden_size) # weights between input and hidden layer
    #     self.bias_input_hidden = np.zeros((1, hidden_size)) # bias between input and hidden layer
    #     self.weights_hidden_output = np.random.randn(hidden_size, output_size) # weights between hidden and output layer
    #     self.bias_hidden_output = np.zeros((1, output_size)) # bias between hidden and output layer
        
    def initialize_weights(self, input_size, output_size):
        weights = np.random.randn(input_size, output_size)
        weights = np.clip(weights, -1, 1)  # membatasi rentang inisialisasi [-1, 1]
        return weights
        
    def forward(self, X):
        # Z_hidden = X * W_input_hidden + B_input_hidden
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden # hidden_input -> Z_hidden
        # A_hidden = 1 / (1 + exp(-Z_hidden))
        self.hidden_output = 1 / (1 + np.exp(-self.hidden_input))  # hidden_output -> A_hidden
        # Z_output = A_Hidden * W_hidden_output + B_hidden_output
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output # Output layer -> Z_output
        return self.output 
    
    def backward(self, X, y, learning_rate):
        # d_output = (y^ - y)
        output_error = (self.output.T - y.T).T  # d_output -> output_error
        # d_hidden = d_output * W_hidden_output
        hidden_error = np.dot(output_error, self.weights_hidden_output) # d_hidden -> hidden_error

        # Update bobot dan bias yang menghubungkan hidden layer dan output layer
        # W_hidden_output = W_hidden_output - learning_rate * d_output
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error) 
        # B_hidden_output = B_hidden_output - learning_rate * d_output
        self.bias_hidden_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        
        # Update bobot dan bias yang menghubungkan input layer dan hidden layer
        # W_input_hidden = W_input_hidden - learning_rate * X * d_hidden * A_hidden * (1-A_hidden)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error * self.hidden_output * (1 - self.hidden_output))
        # B_input_hidden = B_input_hidden - learning_rate * d_hidden * A_hidden * (1-A_hidden)
        self.bias_input_hidden -= learning_rate * np.sum(hidden_error * self.hidden_output * (1 - self.hidden_output), axis=0, keepdims=True)
        
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # Small value to prevent division by zero 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip values to prevent division by zero
    # Calculate binary cross entropy (BCE)
    # BCE = -1/N * Σ(y_ij * log(y^_ij) + (1 - y_ij) * log(1 - y^_ij))
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
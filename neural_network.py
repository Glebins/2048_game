import numpy as np


class NeuralNetwork:
    def __init__(self, grid_size):
        self.input_layer_size = grid_size ** 2
        self.hidden_layer_size = 18
        self.output_layer_size = 4
        self.weights_0 = np.random.random((self.input_layer_size, self.hidden_layer_size))
        self.weights_1 = np.random.random((self.hidden_layer_size, self.output_layer_size))

    @classmethod
    def relu(cls, x):
        return x if x > 0 else 0

    @classmethod
    def relu_derivative(cls, x):
        return x > 0

    @classmethod
    def linear(cls, x):
        return x

    def forward(self, input_layer):
        input_layer = np.array([input_layer])
        hidden_layer = self.relu(input_layer @ self.weights_0)
        output_layer = self.linear(hidden_layer @ self.weights_1)

        return np.argmax(output_layer)

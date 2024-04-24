import numpy as np


class NeuralNetwork:
    def __init__(self, grid_size=4, weights=None):
        self.input_layer_size = grid_size ** 2
        self.hidden_layer_size = 18
        self.output_layer_size = 4

        if weights is None:
            self.weights_0 = np.random.random((self.input_layer_size, self.hidden_layer_size))
            self.weights_1 = np.random.random((self.hidden_layer_size, self.output_layer_size))
        else:
            self.weights_0 = weights[0]
            self.weights_1 = weights[1]

    @classmethod
    def relu(cls, x):
        return x * (x > 0)

    @classmethod
    def relu_derivative(cls, x):
        return x > 0

    @classmethod
    def linear(cls, x):
        return x

    def forward(self, input_layer):
        input_layer = np.array([input_layer]).flatten()
        hidden_layer = self.relu(input_layer @ self.weights_0)
        output_layer = self.linear(hidden_layer @ self.weights_1)

        return np.argmax(output_layer)

    @classmethod
    def get_direction(cls, input_neuron_number):
        match input_neuron_number:
            case 0:
                direction = 'Up'
            case 1:
                direction = 'Down'
            case 2:
                direction = 'Left'
            case 3:
                direction = 'Right'
            case _:
                raise Exception("There is no such a direction")

        return direction

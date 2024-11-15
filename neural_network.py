import numpy as np


class NeuralNetwork:
    def __init__(self, grid_size=4, weights=None):
        self.input_layer_size = grid_size ** 2
        self.hidden_layer_size = 50
        self.output_layer_size = 4

        if weights is None:
            self.weights_0 = 2 * np.random.random((self.input_layer_size, self.hidden_layer_size)) - 1
            self.weights_1 = 2 * np.random.random((self.hidden_layer_size, self.output_layer_size)) - 1
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

    @classmethod
    def handle_input(cls, input_layer):
        input_layer[input_layer == 0] += 1
        input_layer = np.log2(input_layer)
        input_layer /= np.max(input_layer)

    def forward(self, input_layer):
        input_layer = np.array([input_layer]).flatten()
        self.handle_input(input_layer)
        hidden_layer = self.relu(input_layer @ self.weights_0)
        output_layer = self.linear(hidden_layer @ self.weights_1)

        return output_layer

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

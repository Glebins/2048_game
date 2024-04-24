import numpy as np
import itertools
import random

from neural_network import NeuralNetwork
from game import Game2048


class IndividualNN:
    def __init__(self, grid_size, border_prob=0.45, weights=None):
        self.grid_size = grid_size
        self.score = -1
        self.chromosome = NeuralNetwork(grid_size, weights)
        self.border_prob = border_prob

    @classmethod
    def create_genome(cls, nn):
        weights_0 = np.random.random((nn.input_layer_size, nn.hidden_layer_size))
        weights_1 = np.random.random((nn.hidden_layer_size, nn.output_layer_size))

        return [weights_0, weights_1]

    @classmethod
    def get_random_gene(cls):
        return np.random.random()

    def crossover(self, parent_2):
        child_chromosomes_0 = np.zeros_like(self.chromosome.weights_0)
        child_chromosomes_1 = np.zeros_like(self.chromosome.weights_1)
        child_chromosomes = [child_chromosomes_0, child_chromosomes_1]
        weights_parent_1 = [self.chromosome.weights_0, self.chromosome.weights_1]
        weights_parent_2 = [parent_2.chromosome.weights_0, parent_2.chromosome.weights_1]

        # it = np.nditer(weights_parent_1, flags=['multi_index'])
        shape = [self.chromosome.weights_0]
        # todo FUCKK

        for position in itertools.product(*[range(s) for s in [2, 3]]):
            pass

        for gp1 in it:
            position = it.multi_index
            gp2 = weights_parent_2[position]

            prob = random.random()

            if prob < self.border_prob:
                child_chromosomes[position] = gp1
            elif self.border_prob <= prob < 2 * self.border_prob:
                child_chromosomes[position] = gp2
            else:
                child_chromosomes[position] = np.append(self.get_random_gene())

        return IndividualNN(self.grid_size, self.border_prob, child_chromosomes)

    def simulate_game(self):
        game_engine = Game2048(self.grid_size)

        while True:
            game_engine.do_move(self.chromosome.get_direction(self.chromosome.forward(game_engine.grid)))

            if game_engine.test_if_the_game_over() or game_engine.prev_grid == game_engine.grid:
                score = game_engine.score
                self.score = score
                return score

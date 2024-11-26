import numpy as np
import random

from neural_network import NeuralNetwork
from game import Game2048


class IndividualNN:
    def __init__(self, grid_size, mutation_coefficient, weights=None):
        self.grid_size = grid_size
        self.score = -1
        self.chromosome = NeuralNetwork(grid_size, weights)
        self.mutation_coefficient = mutation_coefficient

    @classmethod
    def create_genome(cls, nn):
        weights_0 = np.random.random((nn.input_layer_size, nn.hidden_layer_size))
        weights_1 = np.random.random((nn.hidden_layer_size, nn.output_layer_size))

        return [weights_0, weights_1]

    @classmethod
    def get_random_gene(cls):
        return np.random.random()

    @classmethod
    def mate_one_array(cls, parent_1_weights, parent_2_weights):
        child_chromosomes = []

        crossover_point = np.random.randint(1, len(parent_1_weights))
        prob = random.random()

        for i, gp1, gp2 in zip(range(parent_1_weights.size), parent_1_weights.flatten(), parent_2_weights.flatten()):

            prob = random.random()

            if i < crossover_point:
                child_chromosomes.append(gp1)
            else:
                child_chromosomes.append(gp2)

            # if prob < 0.5:
            #     child_chromosomes.append(gp1)
            # else:
            #     child_chromosomes.append(gp2)

        return np.array(child_chromosomes)

    def mutate_one_array(self, child_weights):
        # mutated_array = np.copy(child_weights)
        mutation_mask = np.random.rand(*child_weights.shape) < self.mutation_coefficient
        child_weights[mutation_mask] = np.random.rand(*child_weights.shape)[mutation_mask]
        return child_weights

    def crossover(self, parent_2):
        child_chromosomes_0 = self.mate_one_array(self.chromosome.weights_0, parent_2.chromosome.weights_0)
        child_chromosomes_1 = self.mate_one_array(self.chromosome.weights_1, parent_2.chromosome.weights_1)

        child_chromosomes_0 = child_chromosomes_0.reshape(self.chromosome.weights_0.shape)
        child_chromosomes_1 = child_chromosomes_1.reshape(self.chromosome.weights_1.shape)

        child_chromosomes = [child_chromosomes_0, child_chromosomes_1]

        return IndividualNN(self.grid_size, self.mutation_coefficient, weights=child_chromosomes)

    def mutate(self):
        self.chromosome.weights_0 = self.mutate_one_array(self.chromosome.weights_0)
        self.chromosome.weights_1 = self.mutate_one_array(self.chromosome.weights_1)

    def simulate_game(self):
        game_engine = Game2048(self.grid_size)
        game_engine.place_random_tile()

        while True:
            moves = self.chromosome.forward(game_engine.grid)
            moves = np.argsort(-moves)

            move_i = 0
            game_engine.apply_move(self.chromosome.get_direction(moves[move_i]))

            while game_engine.grid == game_engine.prev_grid:
                move_i += 1
                game_engine.apply_move(self.chromosome.get_direction(moves[move_i]))

            if game_engine.is_game_over:
                score = game_engine.score
                self.score = score
                return score

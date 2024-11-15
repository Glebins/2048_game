from neural_network_genetics import IndividualNN
import random
import numpy as np


class EvolutionNN:
    def __init__(self, grid_size=4, population_size=100, elitism_coefficient=0.1, mating_coefficient=0.5,
                 mutation_coefficient=0.1):
        self.grid_size = grid_size
        self.population_size = population_size
        self.elitism_coefficient = elitism_coefficient
        self.mating_coefficient = mating_coefficient
        self.mutation_coefficient = mutation_coefficient
        self.population = []
        self.generation = 0

        self.create_population()

    def create_population(self):
        for _ in range(self.population_size):
            individual = IndividualNN(self.grid_size, self.mutation_coefficient)
            self.population.append(individual)

    def generation_pass(self):
        self.evaluate_population()
        self.get_offspring()

    def evaluate_population(self):
        self.run_simulations()
        self.population = sorted(self.population, key=lambda x: -x.score)

    def get_offspring(self):
        new_generation = []
        new_generation.extend(self.population[:int(self.population_size * self.elitism_coefficient)])

        for _ in range(int(self.population_size * (1 - self.elitism_coefficient))):
            parent_1 = random.choice(self.population[:int(self.population_size * self.mating_coefficient)])
            parent_2 = random.choice(self.population[:int(self.population_size * self.mating_coefficient)])
            child = parent_1.crossover(parent_2)
            child.mutate()
            new_generation.append(child)

        self.population = new_generation
        self.generation += 1

    def run_simulations(self):
        for p in self.population:
            p.simulate_game()

    def print_generation_info(self):
        print(f"Generation {self.generation}, String = {''.join(self.population[0].chromosome)}, "
              f"Score = {self.population[0].score}")

    def print_scores_of_population(self):
        print(f"{self.generation}: {[x.score for x in self.population]}")

    def save_weights_in_file(self, file_to_save):
        w0 = []
        w1 = []

        n = self.population_size

        for i in range(n):
            w0.append(self.population[i].chromosome.weights_0)
            w1.append(self.population[i].chromosome.weights_1)

        w0 = np.array(w0).flatten()
        w1 = np.array(w1).flatten()

        file = open(file_to_save, "wb")
        np.save(file, w0)
        np.save(file, w1)

        i_s = self.population[0].chromosome.input_layer_size
        h_s = self.population[0].chromosome.hidden_layer_size
        o_s = self.population[0].chromosome.output_layer_size

        params = np.array([n, i_s, h_s, o_s])
        np.save(file, params)

        file.close()

    def read_weights_from_file(self, file_with_weights):
        file = open(file_with_weights, "rb")
        w0 = np.load(file)
        w1 = np.load(file)

        n, i_s, h_s, o_s = np.load(file).tolist()

        w0 = w0.reshape((n, i_s, h_s))
        w1 = w1.reshape((n, h_s, o_s))

        for i in range(n):
            self.population[i].chromosome.weights_0 = w0[i]
            self.population[i].chromosome.weights_1 = w1[i]

        file.close()

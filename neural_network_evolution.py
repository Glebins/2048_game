from neural_network_genetics import IndividualNN
from neural_network import NeuralNetwork
import random


class EvolutionNN:
    def __init__(self, grid_size=4, population_size=100, elitism_coefficient=0.1, mating_coefficient=0.5):
        self.grid_size = grid_size
        self.population_size = population_size
        self.elitism_coefficient = elitism_coefficient
        self.mating_coefficient = mating_coefficient
        self.population = []
        self.generation = 0

        self.create_population()

    def create_population(self):
        for _ in range(self.population_size):
            individual = IndividualNN(self.grid_size)
            self.population.append(individual)
            # individual_weights = IndividualNN.create_genome(NeuralNetwork, self.grid_size)
            # self.population.append(IndividualNN(self.grid_size, weights=individual_weights))

    def generation_pass(self):
        self.run_simulations()
        self.population = sorted(self.population, key=lambda x: x.score)
        print([x.score for x in self.population])

        new_generation = []
        new_generation.extend(self.population[:int(self.population_size * self.elitism_coefficient)])

        for _ in range(int(self.population_size * (1 - self.elitism_coefficient))):
            parent_1 = random.choice(self.population[:int(self.population_size * self.mating_coefficient)])
            parent_2 = random.choice(self.population[:int(self.population_size * self.mating_coefficient)])
            child = parent_1.crossover(parent_2)
            new_generation.append(child)

        self.population = new_generation
        self.generation += 1

    def run_simulations(self):
        for p in self.population:
            print(p.simulate_game())

    def print_generation_info(self):
        print(f"Generation {self.generation}, String = {''.join(self.population[0].chromosome)}, "
              f"Score = {self.population[0].score}")
        # np.savetxt("test_weights.txt", self.population[0].chromosome)

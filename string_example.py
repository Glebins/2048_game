import tkinter as tk

from game import *
from genetics import *

# root = tk.Tk()
# game = Game2048(root)
# root.mainloop()

genes = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

population_size = 1000
target = "blow your mind"
elitism_coefficient = 0.1
mating_coefficient = 0.5

population = []
generation = 1

for _ in range(population_size):
    individual = Genetics.create_genome(genes, target)
    population.append(Genetics(genes, individual))

while population[0].get_fitness(target) > 0:
    population = sorted(population, key=lambda x: x.get_fitness(target))

    new_generation = []
    new_generation.extend(population[:int(population_size * elitism_coefficient)])

    for _ in range(int(population_size * (1 - elitism_coefficient))):
        parent_1 = random.choice(population[:int(population_size * mating_coefficient)])
        parent_2 = random.choice(population[:int(population_size * mating_coefficient)])
        child = parent_1.crossover(parent_2)
        new_generation.append(child)

    population = new_generation

    print(f"Generation {generation}, String = {''.join(population[0].chromosome)}, "
          f"Fitness = {population[0].get_fitness(target)}")

    generation += 1

import time

from game_draw import *
import numpy as np
from neural_network import NeuralNetwork
from neural_network_genetics import IndividualNN
from neural_network_evolution import EvolutionNN
import tkinter as tk

file_to_save = "weights.txt"
grid_size = 4
n = 500

game_evolution = EvolutionNN(grid_size, population_size=500, mating_coefficient=0.5, mutation_coefficient=0.1)

game_evolution.read_weights_from_file(file_to_save)
game_evolution.evaluate_population()

for i in range(n):
    game_evolution.evaluate_population()

    scores = []
    for pop in game_evolution.population:
        scores.append(pop.score)
    print(sum(scores) / len(scores))

    game_evolution.print_scores_of_population()
    game_evolution.get_offspring()

    if i % 100 == 0:
        game_evolution.save_weights_into_file(file_to_save)

game_evolution.save_weights_into_file(file_to_save)

# game_evolution.read_weights_from_file(file_to_save)
# game_evolution.evaluate_population()
#
# scores = []
# for i in game_evolution.population:
#     scores.append(i.score)
# print(sum(scores) / len(scores))
#
# game_evolution.print_scores_of_population()


# genes = ['Right', 'Left', 'Up', 'Down']
#
# root = tk.Tk()
# visual_game = DrawGame2048(root, grid_size)
#
# game_engine = visual_game.Game
# game_engine.place_random_tile()
# nn_game = game_evolution.population[0].chromosome
#
# i = 0
#
# while True:
#     moves = nn_game.forward(game_engine.grid)
#     moves = np.argsort(-moves)
#
#     move_i = 0
#     game_engine.do_move(nn_game.get_direction(moves[move_i]))
#
#     while game_engine.grid == game_engine.prev_grid:
#         move_i += 1
#         game_engine.do_move(nn_game.get_direction(moves[move_i]))
#
#     if game_engine.is_game_over:
#         score = game_engine.score
#         print(f"Exit at {i} with a score of {game_engine.score}")
#         break
#
#     visual_game.update_display()
#     root.update_idletasks()
#     root.update()
#
#     i += 1
#     time.sleep(0.2)

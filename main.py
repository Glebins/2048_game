import time

from game_draw import *
from neural_network import NeuralNetwork
from neural_network_genetics import IndividualNN
from neural_network_evolution import EvolutionNN
import tkinter as tk

grid_size = 4
game_evolution = EvolutionNN(grid_size)

game_evolution.generation_pass()

print(game_evolution.population[0].score)

''' grid_size = 4
genes = ['Right', 'Left', 'Up', 'Down']

root = tk.Tk()
visual_game = DrawGame2048(root, grid_size)

game_engine = visual_game.Game
nn_game = NeuralNetwork(grid_size)

i = 0

while True:
    game_engine.do_move(nn_game.get_direction(nn_game.forward(game_engine.grid)))
    print(i, game_engine.grid)

    if game_engine.test_if_the_game_over():
        print(f"Exit at {i} with a score of {game_engine.score}")
        break

    visual_game.update_display()
    root.update_idletasks()
    root.update()

    i += 1
    time.sleep(0.2) '''

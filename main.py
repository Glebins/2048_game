import time
import tkinter as tk

import numpy as np
import torch
import gymnasium as gym

from expectimax import *
from game_draw import *
from minimax import MinimaxTree

game = Game2048()
# game.place_random_tile()
game.set_state([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]])

tree_game = ExpectimaxTree(game)

depth = 4
tree_game.build_tree(tree_game.root, depth=depth, maximizing_player=True)

print(tree_game.root.state, end='\n\n\n')

node = tree_game.root
is_maximizing = True


step = 0

while not game.is_terminal():
    best_move = tree_game.get_best_move(node, is_maximizing)
    if best_move == -1:
        break

    sug_move, pot_score = best_move
    # if is_maximizing:
    #     print(f"Suggested move to win the game: {sug_move}, best score u can achieve: {pot_score}")
    #     players_move = int(input("Your move:"))
    #     game.apply_move(players_move)
    # else:
    print(f"Suggested move to win the game: {sug_move}, best score AI can achieve: {pot_score}")
    game.apply_move(sug_move)
    step += 1

    for i in node.children:
        if i.state == game.get_state():
            node = i
            break

    # print(sug_move, pot_score)
    print(game.get_state(), game.evaluate())

    is_maximizing = not is_maximizing

    if step % depth == 0:
        tree_game.build_tree(node, depth=depth, maximizing_player=is_maximizing)


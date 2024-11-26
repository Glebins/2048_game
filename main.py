import time
import tkinter as tk

import numpy as np
import torch
import gymnasium as gym

from expectimax import *
from game_draw import *
from minimax import MinimaxTree

game = Game2048()

tree_game = MinimaxTree(game)

tree_game.build_tree(tree_game.root, depth=3, maximizing_player=True)

for child in tree_game.root.children:
    print(child.state)


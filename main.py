import time

from game import *
from game_draw import *
import tkinter as tk

grid_size = 4
genes = ['Right', 'Left', 'Up', 'Down']

root = tk.Tk()
visual_game = DrawGame2048(root, grid_size)

game_engine = visual_game.Game

i = 0

while True:
    game_engine.do_move(random.choice(genes))
    print(i, game_engine.grid)

    if game_engine.test_if_the_game_over():
        print(f"Exit at {i} with a score of {game_engine.score}")
        break

    visual_game.update_display()
    root.update_idletasks()
    root.update()

    i += 1
    time.sleep(0.2)


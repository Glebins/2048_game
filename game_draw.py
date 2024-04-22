import tkinter as tk
from game import *


class DrawGame2048:
    def __init__(self, window, grid_size):
        self.grid_size = grid_size
        self.Game = Game2048(self.grid_size)

        self.tiles = None
        self.score_label = None

        self.window = window
        self.window.title("2048 Game")
        self.window.bind("<Key>", self.key_pressed)

        self.create_widgets()
        self.Game.place_random_tile()
        self.update_display()

    def create_widgets(self):
        self.score_label = tk.Label(self.window, text="Score: 0", font=("Helvetica", 16))
        self.score_label.grid(row=0, column=0, columnspan=self.grid_size)

        self.tiles = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                tile = tk.Label(self.window, text="", font=("Helvetica", 24), width=6, height=3, relief="raised")
                tile.grid(row=i + 1, column=j)
                row.append(tile)
            self.tiles.append(row)

    def update_display(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.Game.grid[i][j]
                text = str(value) if value != 0 else ""
                self.tiles[i][j].config(text=text, bg=self.get_tile_color(value))

        self.score_label.config(text=f"Score: {self.Game.score}")

    @classmethod
    def get_tile_color(cls, value):
        colors = {
            2: "#eee4da",
            4: "#ede0c8",
            8: "#f2b179",
            16: "#f59563",
            32: "#f67c5f",
            64: "#f65e3b",
            128: "#edcf72",
            256: "#edcc61",
            512: "#edc850",
            1024: "#edc53f",
            2048: "#edc22e"
        }
        return colors.get(value, "#cdc1b4")

    def key_pressed(self, event):
        key = event.keysym
        if key in ['Up', 'Down', 'Left', 'Right']:
            self.Game.do_move(key)
            self.update_display()
        elif key == 'b':
            if self.Game.prev_grid is not None:
                self.Game.grid = self.Game.prev_grid
                self.Game.score = self.Game.prev_score
            self.update_display()
        elif key == 'z':
            self.window.destroy()

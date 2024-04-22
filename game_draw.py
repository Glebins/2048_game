import tkinter as tk
import random
import copy


class DrawGame2048:
    def __init__(self, window):
        self.tiles = None
        self.score_label = None
        self.prev_grid = None
        self.prev_score = None

        self.window = window
        self.window.title("2048 Game")
        self.window.bind("<Key>", self.key_pressed)

        self.grid_size = 4
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0

        self.create_widgets()
        self.place_random_tile()
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

    def place_tile(self, x, y, val):
        self.grid[x][y] = val

    def place_random_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row][col] = 2 if random.random() < 0.9 else 4

    def update_display(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.grid[i][j]
                text = str(value) if value != 0 else ""
                self.tiles[i][j].config(text=text, bg=self.get_tile_color(value))

        self.score_label.config(text=f"Score: {self.score}")

    def get_tile_color(self, value):
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
            self.prev_grid = copy.deepcopy(self.grid)
            self.prev_score = self.score
            self.move_tiles(key)
            if self.grid != self.prev_grid:
                self.place_random_tile()
                self.update_display()
            if self.is_the_end():
                print("Game over")
        elif key == 'b':
            if self.prev_grid is not None:
                self.grid = self.prev_grid
                self.score = self.prev_score
            self.update_display()
        elif key == 'z':
            self.window.destroy()

    def is_the_end(self):
        now_grid = copy.deepcopy(self.grid)

        for it in ['Up', 'Down', 'Left', 'Right']:
            self.grid = now_grid
            self.move_tiles(it)
            if self.grid != now_grid:
                self.grid = now_grid
                return False
        return True

    def move_tiles(self, direction):
        if direction == 'Up':
            self.transpose()
            self.reverse_rows()
            self.merge_tiles()
            self.reverse_rows()
            self.transpose()
        elif direction == 'Down':
            self.transpose()
            self.merge_tiles()
            self.transpose()
        elif direction == 'Left':
            self.reverse_rows()
            self.merge_tiles()
            self.reverse_rows()
        elif direction == 'Right':
            self.merge_tiles()

    def merge_tiles(self):
        for i in range(self.grid_size):
            last_space = self.grid_size - 1
            prev_number = -1
            j = self.grid_size - 1

            while last_space >= 0 and self.grid[i][last_space] != 0:
                last_space -= 1

            while j >= 0:
                if self.grid[i][j] == 0:
                    j -= 1
                    continue

                if prev_number == self.grid[i][j]:
                    pos = self.define_position_of_merged_tile(i, j)
                    self.grid[i][pos] *= 2
                    self.score += self.grid[i][pos]
                    self.grid[i][j] = 0
                    prev_number = -1

                    last_space = pos - 1
                    j -= 1

                elif last_space > j:
                    self.grid[i][last_space] = self.grid[i][j]
                    prev_number = self.grid[i][j]
                    self.grid[i][j] = 0
                    last_space -= 1
                    j -= 1

                else:
                    prev_number = self.grid[i][j]
                    j -= 1

                # if j > 0 and self.grid[i][j] == self.grid[i][j - 1] and self.grid[i][j] != 0:
                #     self.grid[i][j] *= 2
                #     self.score += self.grid[i][j]
                #     self.grid[i][j - 1] = 0
                #
                #     if last_space > j:
                #         self.grid[i][last_space] = self.grid[i][j]
                #         self.grid[i][j] = 0
                #         last_space -= 1
                #     else:
                #         last_space = j - 1
                #
                #     was_merger = True
                #
                # if last_space == -1:
                #     j -= 1
                #     continue
                #
                # if self.grid[i][last_space] == 0 and self.grid[i][j] != 0 and last_space > j:
                #     self.grid[i][last_space] = self.grid[i][j]
                #     self.grid[i][j] = 0
                #     last_space -= 1
                #
                # if was_merger:
                #     j -= 2
                # else:
                #     j -= 1

    def define_position_of_merged_tile(self, i, j):
        for x in range(j + 1, self.grid_size):
            if self.grid[i][x] == self.grid[i][j]:
                return x

    def reverse_rows(self):
        self.grid = [row[::-1] for row in self.grid]

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

import random
import copy


class Game2048:
    def __init__(self, grid_size=4):
        self.prev_grid = None
        self.prev_score = None
        self.is_game_over = False

        self.grid_size = grid_size
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0

    def place_tile(self, x, y, val):
        self.grid[x][y] = val

    def place_random_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row][col] = 2 if random.random() < 0.9 else 4

    def do_move(self, direction):
        if direction not in ['Up', 'Down', 'Left', 'Right']:
            return

        self.prev_grid = copy.deepcopy(self.grid)
        self.prev_score = self.score
        self.move_tiles(direction)

        if self.grid != self.prev_grid:
            self.place_random_tile()
        else:
            self.score = self.prev_score

        if self.test_if_the_game_over():
            self.is_game_over = True

    def test_if_the_game_over(self):
        now_grid = copy.deepcopy(self.grid)
        current_score = self.score

        for it in ['Up', 'Down', 'Left', 'Right']:
            self.grid = now_grid
            self.move_tiles(it)
            if self.grid != now_grid:
                self.grid = now_grid
                self.score = current_score
                return False

        self.score = current_score
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

    def define_position_of_merged_tile(self, i, j):
        for x in range(j + 1, self.grid_size):
            if self.grid[i][x] == self.grid[i][j]:
                return x

    def reverse_rows(self):
        self.grid = [row[::-1] for row in self.grid]

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

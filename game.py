import math
import random
import copy

import numpy as np


class Game2048:
    def __init__(self, grid_size=4):
        self.prev_grid = None
        self.prev_score = None
        self.is_game_over = False

        self.last_move = 0
        self.number_identical_last_moves = 0

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

    def apply_move(self, direction):
        if direction not in ['Up', 'Down', 'Left', 'Right']:
            return

        if self.last_move == direction:
            self.number_identical_last_moves += 1
        else:
            self.number_identical_last_moves = 0
        self.last_move = direction

        self.prev_grid = copy.deepcopy(self.grid)
        self.prev_score = self.score
        self.move_tiles(direction)

        if self.grid != self.prev_grid:
            self.place_random_tile()
            was_positions_changed = True
        else:
            self.score = self.prev_score
            was_positions_changed = False

        if self.test_if_the_game_over():
            self.is_game_over = True

        return was_positions_changed

    def clear(self):
        self.__init__(self.grid_size)

    def init(self):
        self.clear()
        self.place_random_tile()

    def get_state(self):
        return self.grid

    def set_state(self, new_grid):
        self.grid = new_grid

    def delete_move(self):
        self.grid = self.prev_grid
        self.score = self.prev_score
        self.is_game_over = False

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

    def get_legal_moves(self):
        now_grid = self.grid.copy()
        current_score = self.score

        legal_moves = []

        for it in ['Left', 'Down', 'Right', 'Up']:
            self.grid = now_grid
            self.move_tiles(it)

            if self.grid != now_grid:
                self.grid = now_grid
                self.score = current_score
                legal_moves.append(it)
            else:
                continue

        self.score = current_score
        return legal_moves

    def is_terminal(self):
        if self.test_if_the_game_over():
            return True
        else:
            return False

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
            self.grid = [row[:] for row in self.grid]
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

    def get_monotonicity(self):
        mon_score = 0

        for i in range(4):
            for j in range(3):
                if self.grid[i][j + 1] != self.grid[i][j]:
                    mon_score += 1

        for j in range(4):
            for i in range(3):
                if self.grid[i + 1][j] != self.grid[i][j]:
                    mon_score += 1

        return mon_score

    def get_number_empty_cells(self):
        empty_cells = 0
        for i in self.grid:
            if i == 0:
                empty_cells += 1

        return empty_cells

    def get_smoothness(self):
        smoothness = 0

        for i in range(4):
            for j in range(4):
                val = self.grid[i][j] if self.grid[i][j] != 0 else 1
                val = math.log2(val)

                potential_neighbors = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]

                for x, y in potential_neighbors:
                    if x < 0 or x > 3 or y < 0 or y > 3:
                        continue
                    neighbor_val = math.log2(self.grid[x][y]) if self.grid[x][y] != 0 else 1
                    smoothness -= abs(val - neighbor_val)

        return smoothness

    def get_sum_elements(self):
        return np.sum(np.array(self.grid))

    def evaluate(self):
        monotonicity_score = self.get_monotonicity()
        empty_cells = self.get_number_empty_cells()
        smoothness_score = self.get_smoothness()
        sum_elems = self.get_sum_elements()

        monotonicity_weight = 1.0
        emptiness_weight = 2.8
        smoothness_weight = 0.2
        sum_weight = 0.1
        repeatability_weight = 100

        return (monotonicity_score * monotonicity_weight + empty_cells * emptiness_weight +
                smoothness_score * smoothness_weight + math.log2(self.score + 1) + sum_weight * sum_elems -
                repeatability_weight * self.number_identical_last_moves)

    @classmethod
    def simulate_random_game(cls, game):
        genes = ['Right', 'Left', 'Up', 'Down']
        game.place_random_tile()
        while True:
            game.apply_move(random.choice(genes))

            if game.test_if_the_game_over():
                # print(game.grid)
                return game.grid, game.score

import time
import tkinter as tk

import numpy as np
import torch

from DQN import *
from game_draw import *

game = Game2048()

step = 0
counter_bad_moves = 0
game.init()
state = game.get_state()

model = DQN([0, 1, 2, 3], in_dim=16, gamma=1, lr=0.01)

amount_of_epochs = 100_000_000

for epoch in range(amount_of_epochs):
    step += 1
    action = model.action(torch.Tensor(state), 0.1)

    action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

    prev_score = game.score
    was_positions_changed = game.do_move(action_to_do[action])
    new_state = game.get_state()
    reward = game.score - prev_score
    terminated = game.is_game_over

    counter_bad_moves += int(not was_positions_changed)

    if was_positions_changed:
        counter_bad_moves = 0

    # if not was_positions_changed:
    #     reward = 0.9 * reward

    truncated = counter_bad_moves >= 10

    model.add_exp(state, action, reward, new_state, terminated)
    state = new_state

    if terminated or truncated:
        game.init()
        counter_bad_moves = 0

    if step > 100:
        model.update()

    if step % 10_000 == 0:
        test_rewards = model.test_2048(game, 100)
        check_mean = sum(test_rewards) / len(test_rewards)
        print(f"{step}: {check_mean}")


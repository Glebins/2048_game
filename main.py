import time
import tkinter as tk

import numpy as np
import torch

from DQN import *
from game_draw import *

game = Game2048()

step = -1
game.init()
state = game.get_state()

model = DQN([0, 1, 2, 3], in_dim=16, gamma=1, lr=1)

# model.model.load_state_dict(torch.load(f"model_mlp_2700_0.pth"))

amount_of_epochs = 100_000_000

for epoch in range(amount_of_epochs):
    step += 1
    actions = model.actions_by_probability(torch.Tensor(state), 0).detach().numpy()

    action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

    was_positions_changed = False
    prev_score = game.score

    i = 0
    terminated = False

    while not was_positions_changed:
        was_positions_changed = game.do_move(action_to_do[actions[i]])

        new_state = game.get_state()
        reward = game.score - prev_score
        terminated = game.is_game_over

        if not was_positions_changed:
            reward = -100

        model.add_exp(state, actions[i], reward, new_state, terminated)
        i += 1

    state = game.get_state()

    if terminated:
        game.init()
        counter_bad_moves = 0

    if step > 100:
        model.update()

    if step % 2_000 == 0:
        test_rewards, num_steps = np.array(model.test_2048(game, repeats=100, penalty=0))
        print(f"{step}: {test_rewards.mean()},\tmedian = {np.median(test_rewards)},"
              f"\tmin = {test_rewards.min()},\tmax = {test_rewards.max()},"
              f"\tsteps mean = {num_steps.mean()}")
        if test_rewards.mean() > 3100:
            torch.save(model.model.state_dict(), f"model_mlp_2700_1.pth")


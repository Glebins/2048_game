import time
import tkinter as tk
from logging import critical

import numpy as np
import torch
import gymnasium as gym

from ActorCritic import *
from game_draw import *

game = Game2048()

# model = DQN([0, 1, 2, 3], in_dim=16, gamma=1, lr=0.01)

model = Actor_Critic([0, 1, 2, 3], gamma=1, lr_a=0.01, lr_c=0.01, input_size=16, output_size=4, l2_norm=0)

# model.model.load_state_dict(torch.load(f"model_mlp_2700_0.pth"))

epoch = -1
amount_of_epochs = 50_000_000
train = True

while train and epoch < amount_of_epochs:
    epoch += 1

    states, actions, rewards = [], [], []
    game.init()
    state = game.get_state()
    terminated = truncated = False

    time_step = 0
    prev_x_position = state[0]

    while not (terminated or truncated):
        actions_i = model.actions_by_probability(torch.Tensor(state)).detach().numpy()[0]
        action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

        was_positions_changed = False
        prev_score = game.score

        i = 0

        while not was_positions_changed:
            was_positions_changed = game.do_move(action_to_do[actions_i[i]])

            reward = game.score - prev_score
            terminated = game.is_game_over

            if not was_positions_changed:
                reward = -100

            states.append(state)
            actions.append(actions_i[i])
            rewards.append(reward)
            i += 1

        if terminated:
            game.init()
            counter_bad_moves = 0

        state = game.get_state()

    model.update(states, actions, rewards, state, terminated)

    if epoch % 50 == 0:
        test_rewards, num_steps = np.array(model.test_2048(game, repeats=100, penalty=0))
        ac_loss = round(float(model.last_actor_loss), 5)
        cr_loss = round(float(model.last_critic_loss), 5)
        print(f"{epoch}: {test_rewards.mean()}, median = {np.median(test_rewards)},"
              f" min = {test_rewards.min()}, max = {test_rewards.max()},"
              f" steps mean = {num_steps.mean()}, actor loss = {ac_loss}, critic loss = {cr_loss}")
        if test_rewards.mean() > 3100:
            torch.save(model.actor.state_dict(), f"model_actor_3100_1.pth")
            torch.save(model.critic.state_dict(), f"model_critic_3100_1.pth")





# for epoch in range(amount_of_epochs):
#     step += 1
#     actions = model.actions_by_probability(torch.Tensor(state), 0).detach().numpy()
#
#     action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
#
#     was_positions_changed = False
#     prev_score = game.score
#
#     i = 0
#     terminated = False
#
#     while not was_positions_changed:
#         was_positions_changed = game.do_move(action_to_do[actions[i]])
#
#         new_state = game.get_state()
#         reward = game.score - prev_score
#         terminated = game.is_game_over
#
#         if not was_positions_changed:
#             reward = -100
#
#         model.add_exp(state, actions[i], reward, new_state, terminated)
#         i += 1
#
#     state = game.get_state()
#
#     if terminated:
#         game.init()
#         counter_bad_moves = 0
#
#     if step > 100:
#         model.update()
#
#     if step % 2_000 == 0:
#         test_rewards, num_steps = np.array(model.test_2048(game, repeats=100, penalty=0))
#         print(f"{step}: {test_rewards.mean()},\tmedian = {np.median(test_rewards)},"
#               f"\tmin = {test_rewards.min()},\tmax = {test_rewards.max()},"
#               f"\tsteps mean = {num_steps.mean()}")
#         if test_rewards.mean() > 3100:
#             torch.save(model.model.state_dict(), f"model_mlp_2700_1.pth")


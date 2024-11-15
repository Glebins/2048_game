import copy
from itertools import count
from math import trunc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import ReplayBuffer


class Q(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q, self).__init__()
        self.embed = nn.Embedding(16, 32)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def handle_input(self, x):
        print(x.shape)
        x_hat = copy.deepcopy(x)
        x_hat[x_hat == 0] += 1
        x_hat = np.log2(x_hat)
        x_hat /= torch.max(x_hat)
        return x_hat

    def forward(self, x):
        x = torch.log2(x).clamp(min=0).long()  # Tile values like 2, 4, 8, etc. become 1, 2, 3
        x = self.embed(x)

        if x.ndim == 2:
            x = x[None, :]

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 64, 5, 5)
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 128, 5, 5)

        # Flatten the output
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 128 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_vals = self.fc3(x)

        # x = self.handle_input(x)
        # x = F.relu(self.linear1(x.float()))
        # q_vals = self.linear2(x)
        return q_vals


class DQN:
    def __init__(self, actions, in_dim=-1, gamma=1, lr=0.01):
        self.in_dim = in_dim
        self.actions = actions
        self.action_to_ind = {a: i for i, a in enumerate(self.actions)}
        self.gamma = gamma
        self.out_dim = len(actions)
        self.model = Q(self.in_dim, self.out_dim)
        self.buffer = ReplayBuffer(batch_size=100)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.rng = np.random.default_rng()

    def action(self, state, soft=0):
        if self.rng.random() < soft:
            return self.rng.choice(self.actions)
        return self.actions[torch.argmax(self.model(state)).item()]

    def add_exp(self, state, action, reward, new_state, terminated):
        self.buffer.extend([[state, self.action_to_ind[action], reward, new_state, np.int32(terminated)]])

    def update(self):
        state, action, reward, new_state, terminated = self.buffer.sample()[0]
        with torch.no_grad():
            v_max = self.model(new_state).gather(-1, torch.argmax(self.model(new_state), axis=-1, keepdims=True))
            y = reward.ravel() + self.gamma * (1 - terminated) * v_max.ravel()
        pred = self.model(state).gather(-1, action.reshape(-1, 1)).ravel()
        loss = F.mse_loss(y.float(), pred.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, env):
        state, _ = env.reset()
        action = self.action(torch.Tensor(state))
        state, reward, terminated, truncated, info = env.step(action)
        total_reward = reward
        while not(terminated or truncated):
            action = self.action(torch.Tensor(state))
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        return total_reward

    def test_2048(self, game, repeats=100):
        game_copy = copy.deepcopy(game)

        rewards = []

        for i in range(repeats):
            game_copy.init()
            state = game_copy.get_state()

            action = self.action(torch.Tensor(state))
            action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

            prev_score = game_copy.score
            was_positions_changed = game_copy.do_move(action_to_do[action])
            state = game_copy.get_state()
            total_reward = game_copy.score - prev_score
            terminated = game_copy.is_game_over

            counter_bad_moves = int(not was_positions_changed)
            truncated = counter_bad_moves >= 10

            while not (terminated or truncated):
                action = self.action(torch.Tensor(state))

                prev_score = game_copy.score
                was_positions_changed = game_copy.do_move(action_to_do[action])
                state = game_copy.get_state()
                reward = game_copy.score - prev_score
                terminated = game_copy.is_game_over

                total_reward += reward

                if was_positions_changed:
                    counter_bad_moves = 0

                counter_bad_moves += int(not was_positions_changed)
                truncated = counter_bad_moves >= 10

            rewards.append(total_reward)

        return rewards


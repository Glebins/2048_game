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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)

        # self.linear_1 = nn.Linear(16, 40)
        # self.linear_2 = nn.Linear(40, 4)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(4, 4)
        else:
            x = x.reshape(-1, 4, 4)
        x = torch.log2(x).clamp(min=0).float()
        x = x / x.max()

        if x.ndim == 2:
            x = x[None, None, :]
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        else:
            raise ValueError(f"Strange x shape: {x.ndim} dimensions (should be 2 or 3), "
                             f"shape = {x.shape}")

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_vals = self.fc3(x)

        # x[x == 0] += 1
        # x = torch.log2(x)
        #
        # x = F.relu(self.linear_1(x))
        # q_vals = self.linear_2(x)
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

    def actions_by_probability(self, state, soft=0):
        if self.rng.random() < soft:
            # return self.rng.choice(self.actions)
            return torch.Tensor(self.rng.choice([0, 1, 2, 3], 4, replace=False))
        # return self.actions[torch.argmax(self.model(state)).item()]
        return torch.argsort(-self.model(state))[0]

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
        action = self.actions_by_probability(torch.Tensor(state))
        state, reward, terminated, truncated, info = env.step(action)
        total_reward = reward
        while not(terminated or truncated):
            action = self.actions_by_probability(torch.Tensor(state))
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        return total_reward

    def test_2048(self, game, repeats=100, penalty=0):
        game_copy = copy.deepcopy(game)

        rewards = []
        number_of_steps = []

        for rep in range(repeats):
            game_copy.init()
            state = game_copy.get_state()

            actions = self.actions_by_probability(torch.Tensor(state)).detach().numpy()
            action_to_do = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

            was_positions_changed = False
            terminated = False
            prev_score = game_copy.score
            i = 0
            total_reward = 0

            while not was_positions_changed:
                was_positions_changed = game_copy.do_move(action_to_do[actions[i]])
                total_reward = game_copy.score - prev_score
                terminated = game_copy.is_game_over

                if not was_positions_changed:
                    total_reward += penalty
                i += 1

            number_of_steps.append(1)

            state = game_copy.get_state()

            while not terminated:
                actions = self.actions_by_probability(torch.Tensor(state)).detach().numpy()

                was_positions_changed = False
                i = 0
                prev_score = game_copy.score

                while not was_positions_changed:
                    was_positions_changed = game_copy.do_move(action_to_do[actions[i]])
                    reward = game_copy.score - prev_score

                    if not was_positions_changed:
                        reward = penalty

                    terminated = game_copy.is_game_over
                    total_reward += reward
                    i += 1

                number_of_steps[rep] += 1

                state = game_copy.get_state()

            number_of_steps[rep] -= 1
            rewards.append(total_reward)

        return rewards, np.array(number_of_steps)


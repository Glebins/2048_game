import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np
import copy

def sample_without_replacement(probs, num_samples):
    probs = probs / probs.sum()
    log_probs = probs.log()

    # Add Gumbel noise
    gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
    perturbed = log_probs + gumbel_noise

    # Select the top `num_samples` indices
    top_indices = perturbed.topk(num_samples).indices
    return top_indices

class Actor(nn.Module):

    def __init__(self, input_size=3, output_size=2):
        super(Actor, self).__init__()
        # self.linear = nn.Linear(input_size, 50)
        # self.hidden = nn.Linear(40, 32)
        # self.actor = nn.Linear(50, output_size)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x[x == 0] += 1
        x = torch.log2(x)
        x /= torch.max(x)

        # x = F.relu(self.linear(x))
        # x = F.relu(self.hidden(x))
        # proba = F.softmax(self.actor(x), dim=-1)

        if x.ndim == 1:
            x = x.reshape(4, 4)
        else:
            x = x.reshape(-1, 4, 4)

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
        proba = F.softmax(self.fc3(x))

        return proba


class Critic(nn.Module):

    def __init__(self, input_size=3):
        super(Critic, self).__init__()
        # self.linear = nn.Linear(input_size, 50)
        # self.hidden_layer = nn.Linear(64, 32)
        # self.critic = nn.Linear(50, 1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x[x == 0] += 1
        x = torch.log2(x)

        # x = F.relu(self.linear(x.float()))
        # x = F.relu(self.hidden_layer(x))
        # v_val = self.critic(x)

        if x.ndim == 1:
            x = x.reshape(4, 4)
        else:
            x = x.reshape(-1, 4, 4)

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
        v_val = self.fc3(x)

        return v_val


class Actor_Critic:

    def __init__(self, actions, gamma=1, lr_a=1e-4, lr_c=1e-4, input_size=3, output_size=2, l2_norm=0):
        self.actions = actions
        self.action_to_ind = {a: i for i, a in enumerate(self.actions)}
        self.gamma = gamma
        self.actor = Actor(input_size, output_size)
        self.critic = Critic(input_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a, weight_decay=l2_norm)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=l2_norm)
        self.critic_optimizer.zero_grad()
        self.last_step = 0

        self.last_actor_loss = None
        self.last_critic_loss = None

    def actions_by_probability(self, state):
        probs = self.actor(state).detach()
        return sample_without_replacement(probs, 4)
        # return self.actions[dist.Categorical(self.actor(state)).sample()]

    def update(self, states, actions, rewards, last_state, terminated):
        self.last_step += len(states)

        cum_rewards = []
        G = (1 - terminated) * self.critic(torch.tensor(last_state)).item()
        for r in reversed(rewards):
            G = r + self.gamma * G
            cum_rewards.append(G)
        cum_rewards = list(reversed(cum_rewards))

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor([self.action_to_ind[a] for a in actions], dtype=torch.float32)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float32)

        pred = self.critic(states).ravel()


        td = cum_rewards - pred
        tmp_ac = dist.Categorical(self.actor(states)).log_prob(actions).ravel() * td.detach().ravel()
        loss_actor = -1 * tmp_ac.sum()
        self.last_actor_loss = loss_actor / tmp_ac.size(0)
        loss_actor.backward()

        # ---------------------------------

        # right_labels = torch.Tensor([model.action(s) for s in states])
        # right_labels = right_labels.long()
        #
        # criterion_loss = nn.CrossEntropyLoss()
        # loss_actor = criterion_loss(self.actor(states), right_labels)
        # self.last_actor_loss = loss_actor
        # loss_actor.backward()

        # --------------------------------

        # right_labels = torch.Tensor([model_2.action(s) for s in states])
        # right_labels = right_labels.long()
        #
        # criterion_loss = nn.CrossEntropyLoss()
        # loss_actor = criterion_loss(self.actor(states), right_labels)
        # self.last_actor_loss = loss_actor
        # loss_actor.backward()


        loss_critic = F.mse_loss(cum_rewards.float(), pred.float())
        self.last_critic_loss = loss_critic / pred.size(0)
        loss_critic.backward()

        if self.last_step > 200:
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            self.last_step = 0

    def test(self, env):
        state, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        while not (terminated or truncated):
            action = self.actions_by_probability(torch.Tensor(state))[0]
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

            actions = self.actions_by_probability(torch.Tensor(state)).detach().numpy()[0]
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
                actions = self.actions_by_probability(torch.Tensor(state)).detach().numpy()[0]

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


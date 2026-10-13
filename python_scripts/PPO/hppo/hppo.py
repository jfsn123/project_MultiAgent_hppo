import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from python_scripts.PPO.utils.Project_config import device


def _to_clean_tensor(data, *, dtype=torch.float32):
    tensor = torch.as_tensor(data, dtype=dtype, device=device)
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _min_max_normalize(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)


class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, num_actions, state_dim=20, use_image_input=True):
        super().__init__()
        self.num_actions = int(num_actions)
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)

        self.relu = nn.ReLU()
        if self.use_image_input:
            self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=(2, 2), padding=1)
            self.conv2 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2))
            self.conv3 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2), padding=1)
            self.fc0 = nn.Linear(6272, 1000)
            self.fc1 = nn.Linear(1000, 100)

        self.fc2 = nn.Identity()
        self.fc4 = nn.Linear(100 + self.state_dim, 200)
        self.discrete_head = nn.Linear(200, self.num_actions)
        self.critic = nn.Linear(200, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _fit_state_dim(self, x):
        x = x.reshape(-1)
        if x.numel() == self.state_dim:
            return x
        if x.numel() > self.state_dim:
            return x[: self.state_dim]
        pad = torch.zeros(self.state_dim - x.numel(), dtype=x.dtype, device=x.device)
        return torch.cat((x, pad), dim=0)

    def _image_features(self, x):
        if not self.use_image_input or x is None:
            return torch.zeros(100, dtype=torch.float32, device=device)

        x = _to_clean_tensor(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1).flatten()
        return _min_max_normalize(self.fc1(self.fc0(x)))

    def _state_features(self, x, layer1, layer2=None):
        x = self._fit_state_dim(_to_clean_tensor(x))
        x = layer1(x)
        if layer2 is not None:
            x = layer2(x)
        return _min_max_normalize(x)

    def forward(self, x, state, x_graph=None):
        image_features = self._image_features(x)
        state_features = self._state_features(state, self.fc2)

        fused = torch.cat((image_features, state_features), dim=-1)
        features = self.fc4(fused)

        logits = self.discrete_head(features)
        discrete_probs = torch.softmax(logits, dim=-1).clamp(1e-6, 1.0 - 1e-6)
        value = self.critic(features).view(-1)
        return discrete_probs, value


class HPPO:
    def __init__(self, num_servos, state_dim=20, use_image_input=True, num_actions=2):
        self.num_servos = int(num_servos)
        self.num_actions = int(num_actions)
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_update_epochs = 10
        self.value_coef = 0.5
        self.entropy_coef = 0.1
        self.lr = 2e-4
        self.lr_decay = 0.999

        self.policy = MultiDiscreteActorCritic(
            self.num_actions,
            state_dim=self.state_dim,
            use_image_input=self.use_image_input,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        self.clear_buffer()

    def _split_obs(self, obs, x_graph=None):
        if isinstance(obs, (tuple, list)) and len(obs) >= 2:
            img_input = obs[0]
            state_input = obs[1]
        else:
            img_input = None
            state_input = obs
        return img_input, state_input

    def choose_action(self, obs, x_graph=None, greedy=False):
        with torch.no_grad():
            img_input, state_input = self._split_obs(obs, x_graph)
            discrete_probs, value = self.policy(img_input, state_input)
            dist = Categorical(probs=discrete_probs)

            if greedy:
                actions = torch.argmax(discrete_probs, dim=-1).reshape(-1)
            else:
                actions = dist.sample().reshape(-1)
            log_probs = dist.log_prob(actions).reshape(-1)
            return {
                "discrete_action": actions.cpu().numpy(),
                "discrete_log_prob": log_probs.cpu().numpy(),
                "value": float(value.item()),
            }

    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.values.append(float(value))
        self.log_probs.append(log_prob)
        self.dones.append(bool(done))

    def _bootstrap_value(self):
        if not self.next_states or self.dones[-1] or self.next_states[-1] is None:
            return 0.0
        with torch.no_grad():
            _, value = self.policy(x=self.next_states[-1][0], state=self.next_states[-1][1])
        return float(value.item())

    def calculate_advantages(self):
        advantages = []
        gae = 0.0
        next_value = self._bootstrap_value()
        for i in reversed(range(len(self.rewards))):
            if i < len(self.rewards) - 1:
                next_value = self.values[i + 1]
            non_terminal = 1.0 - float(self.dones[i])
            delta = self.rewards[i] + self.gamma * next_value * non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)
        return torch.as_tensor(advantages, dtype=torch.float32, device=device)

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def learn(self):
        if len(self.states) < 8:
            return 0

        raw_advantages = self.calculate_advantages()
        values = torch.as_tensor(self.values, dtype=torch.float32, device=device)
        returns = raw_advantages + values
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + 1e-8)

        batch_actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device).reshape(-1)
        batch_log_probs = torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device).reshape(-1)

        total_loss = 0.0
        for _ in range(self.policy_update_epochs):
            probs = []
            values_pred = []
            for state in self.states:
                discrete_probs, value = self.policy(state[0], state[1])
                probs.append(discrete_probs.reshape(self.num_actions))
                values_pred.append(value.reshape(1))

            probs = torch.stack(probs)
            values_pred = torch.cat(values_pred)
            dist = Categorical(probs=probs)
            new_log_probs = dist.log_prob(batch_actions)
            ratio = torch.exp(new_log_probs - batch_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values_pred, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += float(loss.item())

        self.scheduler.step()
        self.clear_buffer()
        return total_loss / self.policy_update_epochs

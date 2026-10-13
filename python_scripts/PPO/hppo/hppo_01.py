import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from python_scripts.PPO.utils.Project_config import device



class MultiDiscreteActorCritic(nn.Module):
    """混合动作空间 Actor-Critic 网络。

    输入由图像和普通状态向量两部分组成；
    输出包括每个舵机的二分类离散动作分布、连续动作正态分布和值函数估计。
    """

    def __init__(self, num_servos, state_dim=20, use_image_input=True):
        super().__init__()

        self.num_servos = num_servos
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)

        if self.use_image_input:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(1, 32, (5, 5), stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (5, 5), stride=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 32, (5, 5), stride=(2, 2), padding=1),
            )
            self.image_mlp = nn.Sequential(
                nn.Linear(6272, 1000),
                nn.Linear(1000, 100),
            )

        self.state_encoder = nn.Identity()

        self.fc4 = nn.Linear(100 + self.state_dim, 200)
        self.discrete_head = nn.Linear(200, num_servos * 2)
        self.continuous = nn.Sequential(
            nn.Linear(200, num_servos),
            nn.Tanh(),
        )

        self.actor_log_sigma = nn.Parameter(torch.zeros(num_servos) * 0.1)
        self.critic = nn.Sequential(nn.Linear(200, 1))

    @staticmethod
    def _min_max_normalize(x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        return (x - min_val) / (max_val - min_val + 1e-8)

    def _fit_state_dim(self, x):
        """将输入向量裁剪或补零到 state_dim。"""
        x = x.view(-1)

        if x.numel() == self.state_dim:
            return x
        if x.numel() > self.state_dim:
            return x[: self.state_dim]

        pad = torch.zeros(
            self.state_dim - x.numel(),
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat((x, pad), dim=0)

    def _image_features(self, x):
        """提取图像特征；关闭图像输入时返回同维度零向量。"""
        if not self.use_image_input:
            return torch.zeros(100, dtype=torch.float32, device=device)

        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, dim=0)
        x = self.image_encoder(x)
        x = torch.flatten(x.view(x.size(0), -1))
        return self._min_max_normalize(self.image_mlp(x))

    def _vector_features(self, x, encoder):
        """提取普通状态特征，保持原始状态维度。"""
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = encoder(self._fit_state_dim(x))
        return self._min_max_normalize(x)

    def _shared_features(self, x, state, x_graph=None):
        """融合图像和状态两路特征。"""
        image_features = self._image_features(x)
        state_features = self._vector_features(state, self.state_encoder)

        fused = torch.cat((image_features, state_features), dim=-1)
        return self.fc4(fused)

    def _action_distributions(self, features):
        """根据共享特征构造离散和连续动作分布。"""
        discrete_logits = self.discrete_head(features)
        discrete_logits = discrete_logits.view(-1, self.num_servos, 2)
        discrete_dist = Categorical(probs=F.softmax(discrete_logits, dim=-1))

        mu = self.continuous(features)
        sigma = torch.exp(self.actor_log_sigma.expand_as(mu))
        continuous_dist = Normal(mu, sigma)

        return discrete_dist, continuous_dist

    def forward(self, x, state, x_graph=None):
        features = self._shared_features(x, state, x_graph)
        discrete_dist, continuous_dist = self._action_distributions(features)
        value = self.critic(features)
        return discrete_dist, continuous_dist, value


class HPPO:
    """HPPO 训练器，负责采样动作、缓存经验并执行 PPO 更新。"""

    def __init__(
        self,
        num_servos,
        state_dim=20,
        use_image_input=True,
    ):
        self.num_servos = num_servos
        self.state_dim = int(state_dim)
        self.use_image_input = bool(use_image_input)
        self.device = device

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.policy_update_epochs = 3
        self.value_coef = 0.1
        self.entropy_coef = 0.01
        self.reward_scale = 0.05

        self.lr = 5e-5
        self.lr_decay = 0.995

        self.policy = MultiDiscreteActorCritic(
            num_servos,
            state_dim=self.state_dim,
            use_image_input=self.use_image_input,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_c = torch.optim.Adam(
            list(self.policy.continuous.parameters()) + [self.policy.actor_log_sigma],
            lr=self.lr,
        )
        self.optimizer_d = torch.optim.Adam(
            self.policy.discrete_head.parameters(),
            lr=self.lr,
        )
        self.optimizer_v = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.lr_decay,
        )

        self._clear_buffer()

    def choose_action(self, obs, greedy=False):
        """根据当前观测采样动作，并返回动作、log_prob 和价值估计。

        greedy=True 时使用确定性策略（离散取 argmax，连续取均值），
        适用于测试/评估阶段；训练时使用 greedy=False 保持探索。
        """
        with torch.no_grad():
            discrete_dist, continuous_dist, value = self.policy(
                x=obs[0],
                state=obs[1],
            )

            if greedy:
                discrete_action = torch.argmax(discrete_dist.probs, dim=-1)
            else:
                discrete_action = discrete_dist.sample()
            if discrete_action.dim() > 1:
                discrete_action = discrete_action.squeeze(0)

            if greedy:
                continuous_action = continuous_dist.loc
            else:
                continuous_action = continuous_dist.sample()
            if continuous_action.dim() > 1:
                continuous_action = continuous_action.squeeze(0)

            continuous_action = torch.clamp(
                continuous_action,
                min=-1.0,
                max=1.0,
            )

            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            continuous_log_prob = continuous_dist.log_prob(continuous_action)

            if discrete_log_prob.dim() > 1:
                discrete_log_prob = discrete_log_prob.squeeze(0)
            if continuous_log_prob.dim() > 1:
                continuous_log_prob = continuous_log_prob.squeeze(0)

            return {
                "discrete_action": discrete_action.cpu().numpy(),
                "continuous_action": continuous_action.cpu().numpy(),
                "discrete_log_prob": discrete_log_prob.cpu().numpy(),
                "continuous_log_prob": continuous_log_prob.cpu().numpy(),
                "value": value.item() if isinstance(value, torch.Tensor) else float(value),
            }

    def evaluate_action_log_probs(self, obs, discrete_action, continuous_action):
        with torch.no_grad():
            discrete_dist, continuous_dist, _ = self.policy(
                x=obs[0],
                state=obs[1],
            )

            discrete_action = torch.as_tensor(discrete_action, dtype=torch.long, device=self.device)
            continuous_action = torch.as_tensor(continuous_action, dtype=torch.float32, device=self.device)

            discrete_log_prob = discrete_dist.log_prob(discrete_action)
            continuous_log_prob = continuous_dist.log_prob(continuous_action)

            if discrete_log_prob.dim() > 1:
                discrete_log_prob = discrete_log_prob.squeeze(0)
            if continuous_log_prob.dim() > 1:
                continuous_log_prob = continuous_log_prob.squeeze(0)

            return {
                "discrete_log_prob": discrete_log_prob.cpu().numpy(),
                "continuous_log_prob": continuous_log_prob.cpu().numpy(),
            }

    def store_transition(
        self,
        state,
        discrete_action,
        continuous_action,
        reward,
        next_state,
        done,
        value,
        discrete_log_prob,
        continuous_log_prob,
    ):
        """保存一步交互数据，供后续 PPO 更新使用。"""
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.discrete_log_probs.append(discrete_log_prob)
        self.continuous_log_probs.append(continuous_log_prob)
        self.dones.append(done)

    def calculate_advantages(self):
        """使用 GAE 计算优势函数。"""
        advantages = []
        gae = 0

        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]

            reward = float(self.rewards[i]) * self.reward_scale
            delta = reward + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32).to(device)

    def _clear_buffer(self):
        # 经验缓存字段，与 _clear_buffer 中的属性名保持一致。
        BUFFER_FIELDS = (
            "states",
            "discrete_actions",
            "continuous_actions",
            "rewards",
            "next_states",
            "values",
            "discrete_log_probs",
            "continuous_log_probs",
            "dones",
        )
        """清空经验缓存。"""
        for field in BUFFER_FIELDS:
            setattr(self, field, [])

    def learn(self):
        if len(self.states) < 16:
            return 0, 0

        advantages = self.calculate_advantages()
        old_values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        returns = (advantages + old_values).detach()

        batch_states = self.states
        batch_discrete_actions = torch.as_tensor(
            np.asarray(self.discrete_actions),
            dtype=torch.long,
            device=self.device,
        )
        batch_continuous_actions = torch.as_tensor(
            np.asarray(self.continuous_actions),
            dtype=torch.float32,
            device=self.device,
        )

        # 旧策略的 log_prob 在舵机/动作维度上求和，作为 PPO 概率比的分母。
        batch_discrete_log_probs = torch.as_tensor(
            np.asarray(self.discrete_log_probs),
            dtype=torch.float32,
            device=self.device,
        ).view(len(batch_states), -1).sum(dim=1, keepdim=True)
        batch_continuous_log_probs = torch.as_tensor(
            np.asarray(self.continuous_log_probs),
            dtype=torch.float32,
            device=self.device,
        ).view(len(batch_states), -1).sum(dim=1, keepdim=True)

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        advantages = advantages.unsqueeze(1)

        loss_actor = 0
        loss_critic = 0

        for _ in range(self.policy_update_epochs):
            all_discrete_dists = []
            all_continuous_dists = []
            all_values = []

            for state in batch_states:
                discrete_dist, continuous_dist, value = self.policy(
                    x=state[0],
                    state=state[1],
                )
                all_discrete_dists.append(discrete_dist)
                all_continuous_dists.append(continuous_dist)
                all_values.append(value)

            all_values = torch.cat(all_values)
            new_discrete_log_probs = torch.stack(
                [
                    dist.log_prob(batch_discrete_actions[i]).view(-1).sum()
                    for i, dist in enumerate(all_discrete_dists)
                ]
            ).unsqueeze(1)
            new_continuous_log_probs = torch.stack(
                [
                    dist.log_prob(batch_continuous_actions[i]).view(-1).sum()
                    for i, dist in enumerate(all_continuous_dists)
                ]
            ).unsqueeze(1)

            discrete_ratios = torch.exp(
                new_discrete_log_probs - batch_discrete_log_probs
            )
            discrete_unclipped = discrete_ratios * advantages
            discrete_clipped = torch.clamp(
                discrete_ratios,
                1 - self.clip_ratio,
                1 + self.clip_ratio,
            ) * advantages
            discrete_loss = -torch.min(
                discrete_clipped,
                discrete_unclipped,
            ).mean()

            continuous_ratios = torch.exp(
                new_continuous_log_probs - batch_continuous_log_probs
            )
            continuous_unclipped = continuous_ratios * advantages
            continuous_clipped = torch.clamp(
                continuous_ratios,
                1 - self.clip_ratio,
                1 + self.clip_ratio,
            ) * advantages
            continuous_loss = -torch.min(
                continuous_clipped,
                continuous_unclipped,
            ).mean()

            value_loss = F.smooth_l1_loss(all_values.view(-1), returns.view(-1))
            discrete_entropy = torch.stack(
                [dist.entropy().mean() for dist in all_discrete_dists]
            ).mean()
            continuous_entropy = torch.stack(
                [dist.entropy().mean() for dist in all_continuous_dists]
            ).mean()

            total_loss = (
                discrete_loss
                + continuous_loss
                + self.value_coef * value_loss
                - self.entropy_coef * (discrete_entropy + continuous_entropy)
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            loss_actor += (discrete_loss + continuous_loss).item()
            loss_critic += value_loss.item()

        self._clear_buffer()
        return (
            loss_actor / self.policy_update_epochs,
            loss_critic / self.policy_update_epochs,
        )

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time

from .agent_base import BaseAgent
from .ddpg_utils import (
    Policy,
    Critic,
    ReplayBuffer,
    soft_update_params,
    RNDNetwork,
    Logger,
)
from .ddpg_agent import DDPGAgent
from pathlib import Path
from torch.distributions import MultivariateNormal


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    """
    Base DDPG agent that inherits from BaseAgent in agent_base.py.
    In additon, this agent tries the following:
        - Twin delayed DDPG (TD3)       [enabled]
        - RND intrinsic reward          [disabled]
        - Observation normalization     [disabled]
        - Reward shaping                [disabled]
    """

    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device
        self.name = "ddpg"

        # Environment parameters
        self.state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action

        # Training parameters
        self.lr = float(self.cfg.lr)
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau

        # Replay buffer parameters
        self.buffer_size = self.cfg.buffer_size
        self.buffer_ptr = 0
        self.buffer_head = 0
        self.random_transition = 5000  # collect 5k random data for better exploration
        self.max_episode_steps = self.cfg.max_episode_steps

        # Initialize experience buffer
        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, max_size=self.buffer_size
        )

        # Actor
        self.pi = Policy(self.state_dim, self.action_dim, self.max_action).to(
            self.device
        )
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

        # Critic
        self.q = Critic(self.state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        ######### TD3 #########

        # Twin-critic
        self.q2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.lr)

        # Target policy smoothing regularization
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        ######### RANDOM NETWORK DISTILLATION #########
        """
        # Initialize running mean and standard deviation for intrinsic rewards
        self.alpha = 0.01
        self.intrinsic_reward_mean = 0
        self.intrinsic_reward_std = 1

        # Initialize the RND target and predictor networks
        self.rnd_target_dim = 2
        self.rnd_target = RNDNetwork(self.state_dim, self.rnd_target_dim).to(
           self.device
        )
        self.rnd_predictor = RNDNetwork(self.state_dim, self.rnd_target_dim).to(
           self.device
        )
        self.rnd_predictor_optim = torch.optim.Adam(
           self.rnd_predictor.parameters(), lr=self.lr
        )
        """

        ######### OBSERVATION NORMALIZATION ########
        """
        # Observation normalization parameters
        self.alpha = 0.01
        self.observation_mean = 0
        self.observation_std = 52
        """

    def get_intrinsic_reward(self, observation):
        """
        Calculate the RND loss
        """
        # Get RND target and predictor
        rnd_target = self.rnd_target(observation)
        rnd_predictor = self.rnd_predictor(observation)
        rnd_loss = F.mse_loss(rnd_target, rnd_predictor, reduction="none").mean(dim=1)

        return rnd_loss

    def get_normalized_intrinsic_reward(self, intrinsic_reward):
        """
        Normalize the intrinsic reward using running mean and std
        """
        # Update mean
        self.intrinsic_reward_mean = (
            1 - self.alpha
        ) * self.intrinsic_reward_mean + self.alpha * intrinsic_reward.mean()

        # Update std
        self.intrinsic_reward_std = np.sqrt(
            (1 - self.alpha) * (self.intrinsic_reward_std**2)
            + self.alpha * ((intrinsic_reward - self.intrinsic_reward_mean) ** 2).mean()
        )

        # Normalize the intrinsic reward
        intrinsic_reward -= self.intrinsic_reward_mean

        # Clip the reward between 0, 0.3
        intrinsic_reward = np.clip(intrinsic_reward, 0, 0.3)

        return intrinsic_reward.unsqueeze(1)

    def get_normalized_observation(self, observation):
        """
        Normalize the observation using mean and std
        """
        observation -= self.observation_mean
        observation /= self.observation_std

        return observation

    def compute_reward(self, state):
        """
        Compute proximity based rewards.
        Reward for being close to sanding areas and
        penalize for being close to no-sanding areas.
        """
        n_spots = (self.state_dim - 2) // 2
        robot_x, robot_y = state[0:2]
        sanding_areas = state[2 : (2 + n_spots)]
        no_sand_areas = state[(2 + n_spots) :]

        # Distance to sanding and no-sand areas
        d_sanding = [
            np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
            for x, y in zip(sanding_areas[::2], sanding_areas[1::2])
        ]
        d_no_sand = [
            np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
            for x, y in zip(no_sand_areas[::2], no_sand_areas[1::2])
        ]

        # Compute reward
        reward = (sum(d_no_sand) - sum(d_sanding)) / 200
        return reward

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=self.device)

        # Get batch S, A, R, S', D values
        state = batch.state
        action = batch.action  # .to(torch.int64)
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        ### OBSERVATION NORMALIZATION
        """
        next_state = self.get_normalized_observation(next_state)
        """

        ### RND INTRINSIC REWARD
        """
        # Compute RND loss
        rnd_loss = self.get_intrinsic_reward(next_state)

        # Compute intrinsic reward
        intrinsic_reward = rnd_loss.detach()
        intrinsic_reward = self.get_normalized_intrinsic_reward(intrinsic_reward)

        reward += intrinsic_reward

        # Optimize the RND predictor every 10 steps
        if self.buffer_ptr % 10 == 0:
           self.rnd_predictor_optim.zero_grad()
           rnd_loss.mean().backward()
           self.rnd_predictor_optim.step()
        """

        ### REWARD SHAPING
        """
        next_state_np = next_state.detach().numpy()
        reward_shaping = np.array([self.compute_reward(s) for s in next_state_np])
        reward = torch.tensor(reward_shaping, dtype=torch.float).reshape(-1, 1)
        """

        # Computing the target Q-value
        with torch.no_grad():
            # Regularization noise epsilon
            noise = (torch.rand_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            # mu_target(s') + epsilon
            next_action = (self.pi_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Q_i_target(s', mu_target(s'))
            q1_tar = self.q_target(next_state, next_action)
            q2_tar = self.q2_target(next_state, next_action)
            q_tar = torch.min(q1_tar, q2_tar)

            # y(r, s', d) = r + gamma * Q_target(s', mu_target(s')) * (1 - d)
            q_target = reward + self.gamma * q_tar * not_done

        # Update the first critic
        # compute critic loss between Q(s, a) and y(r, s', d)
        q = self.q(state, action)
        critic_loss = F.mse_loss(q, q_target)
        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # Update the second critic
        # compute critic loss between Q(s, a) and y(r, s', d)
        q2 = self.q2(state, action)
        critic_loss2 = F.mse_loss(q2, q_target)
        # optimize the critic
        self.q2_optim.zero_grad()
        critic_loss2.backward()
        self.q2_optim.step()

        # Delayed policy update
        if self.buffer_ptr % self.policy_freq == 0:
            # Compute mu(s)
            mu = self.pi(state)

            # Compute actor loss Q(s, mu(s))
            actor_loss = -self.q(state, mu).mean()

            # Update the actor
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            # Update the target q and target pi using u.soft_update_params() function
            soft_update_params(self.q, self.q_target, self.tau)
            soft_update_params(self.q2, self.q2_target, self.tau)
            soft_update_params(self.pi, self.pi_target, self.tau)

        return {}

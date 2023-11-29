from .agent_base import BaseAgent
from .ddpg_utils import (
    Policy,
    Critic,
    ReplayBuffer,
    soft_update_params,
    DistributionalCritic,
)
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


class DDPGExtension(DDPGAgent):
    """
    Distributed DDPG agent implementation based on
    D4PG (Distributed Distributional Deterministic Policy Gradient)
    """

    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.num_atoms = config.get("num_atoms", 11)
        self.v_min = config.get("v_min", -1)
        self.v_max = config.get("v_max", 1)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = "d4pg"
        self.state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr = float(self.cfg.lr)
        self.buffer_size = self.cfg.buffer_size

        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau

        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0
        self.random_transition = 5000  # collect 5k random data for better exploration
        self.max_episode_steps = self.cfg.max_episode_steps
        # Initialize experience buffer
        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, max_size=self.buffer_size
        )
        # Initizlize policy and critic
        # Actor
        self.pi = Policy(
            self.state_dim,
            self.action_dim,
            self.max_action,
        ).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=self.lr)

        # Critic
        # Replace the critic network with a distributional critic
        self.q = DistributionalCritic(
            self.state_dim,
            self.action_dim,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
        )
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def _update(self):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)

        # Get batch S, A, R, S', D values
        state = batch.state
        action = batch.action.to(torch.int64)
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        # Compute the target distribution
        with torch.no_grad():
            next_action = self.pi_target(next_state)
            next_distribution = F.softmax(
                self.q_target(next_state, next_action), dim=-1
            )
            target_distribution = reward + self.gamma * not_done * next_distribution

        # Update the critic network
        current_distribution = F.log_softmax(self.q(state, action), dim=-1)
        critic_loss = F.kl_div(
            current_distribution, target_distribution, reduction="batchmean"
        )
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # Update the policy network
        policy_loss = -self.q(state, self.pi(state)).mean()
        self.pi_optim.zero_grad()
        policy_loss.backward()
        self.pi_optim.step()

        # Update the target networks
        soft_update_params(self.pi, self.pi_target, self.tau)
        soft_update_params(self.q, self.q_target, self.tau)

        return {"critic_loss": critic_loss.item(), "policy_loss": policy_loss.item()}

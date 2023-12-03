from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer, soft_update_params, Logger

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import copy, time

from pathlib import Path


def to_numpy(tensor):
    """Convert a pytorch tensor to numpy array"""
    return tensor.cpu().numpy().flatten()


class DDPGAgent(BaseAgent):
    """
    Baseline DDPG agent that inherits from BaseAgent in agent_base.py
    """

    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        # ""cuda" if torch.cuda.is_available() else "cpu"
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

        # Initizlize policy and critic

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

    def update(self):
        """
        After collecting one trajectory, update the pi and q for transition times:
        """
        info = {}

        # update the network once per transition
        update_iter = self.buffer_ptr - self.buffer_head

        # update once we have enough data
        if self.buffer_ptr > self.random_transition:
            for _ in range(update_iter):
                info = self._update()

        # update the buffer_head:
        self.buffer_head = self.buffer_ptr

        return info

    def _update(self):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)

        # Get batch S, A, R, S', D values
        state = batch.state
        action = batch.action  # .to(torch.int64)
        next_state = batch.next_state
        reward = batch.reward
        not_done = batch.not_done

        # mu_target(s')
        next_action = self.pi_target(next_state)

        # Q_target(s', mu_target(s'))
        q_tar = self.q_target(next_state, next_action)

        # y(r, s', d) = r + gamma * Q_target(s', mu_target(s')) * (1 - d)
        q_target = reward + self.gamma * q_tar * not_done

        # compute current Q(s, a)
        q = self.q(state, action)

        # compute critic loss between Q(s, a) and y(r, s', d)
        critic_loss = F.mse_loss(q, q_target)

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # Compute mu(s)
        mu = self.pi(state)

        # compute actor loss Q(s, mu(s))
        actor_loss = -self.q(state, mu).mean()

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and target pi using u.soft_update_params() function
        soft_update_params(self.q, self.q_target, self.tau)
        soft_update_params(self.pi, self.pi_target, self.tau)

        return {}

    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        """ "
        Return action for given state as per current policy.
        If evaluation is True, use the greedy action.
        """

        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension

        # convert the observation to torch tensor
        x = torch.from_numpy(observation).float().to(self.device)

        # the stddev of the expl_noise if not evaluation
        expl_noise = 0.05 * self.max_action

        # get the action
        with torch.no_grad():
            action = self.pi(x).squeeze(0)

        if not evaluation:
            # collect random trajectories for better exploration.
            if self.buffer_ptr < self.random_transition:
                # Random actions of shape (action_dim,)
                action = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
                return action, {}

            # Create a multivariate normal distribution
            m = MultivariateNormal(
                torch.zeros(action.shape), torch.eye(action.shape[0]) * expl_noise**2
            )
            # Shape of noise: (action_dim,)
            noise = m.sample()
            action += noise

        # Clip the action within the bounds of the action space
        action = action.clamp(-self.max_action, self.max_action)
        return action, {}  # just return a positional value

    def train_iteration(self):
        """Perform one iteration of learning"""
        # start = time.perf_counter()

        # Run actual training
        reward_sum, timesteps, done = 0, 0, False

        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()

        while not done:
            # TODO: Sample action from policy
            action = self.get_action(obs)[0]

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)
            done_bool = float(done) if timesteps < self.max_episode_steps else 0
            self.record(obs, action, next_obs, reward, done_bool)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

            if timesteps >= self.max_episode_steps:
                done = True

            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        # s = time.perf_counter()
        info = self.update()
        # e = time.perf_counter()

        # Return stats of training
        info.update(
            {
                "episode_length": timesteps,
                "ep_reward": reward_sum,
            }
        )

        return info

    def train(self):
        """Main function to call for training"""
        # create a simple logger to record stats
        if self.cfg.save_logging:
            L = Logger()

        start = time.perf_counter()
        total_step = 0
        run_episode_reward = []
        log_count = 0

        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({"episodes": ep})
            total_step += train_info["episode_length"]
            train_info.update({"total_step": total_step})
            run_episode_reward.append(train_info["ep_reward"])

            # log the training information
            if total_step > self.cfg.log_interval * log_count:
                average_return = sum(run_episode_reward) / len(run_episode_reward)
                if not self.cfg.silent:
                    print(
                        f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}"
                    )
                if self.cfg.save_logging:
                    train_info.update({"average_return": average_return})
                    L.log(**train_info)
                run_episode_reward = []
                log_count += 1

        # save the model weights
        if self.cfg.save_model:
            self.save_model()

        # save the logging information
        logging_path = str(self.logging_dir) + "/logs"
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)

        self.env.close()

        end = time.perf_counter()
        train_time = (end - start) / 60
        print("------ Training Finished ------")
        print(f"Total traning time is {train_time}mins")

    def record(self, state, action, next_state, reward, done):
        """Save transitions to the buffer."""
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    def load_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir) + "/model_parameters_" + str(self.seed) + ".pt"

        d = torch.load(filepath)
        self.q.load_state_dict(d["q"])
        self.q_target.load_state_dict(d["q_target"])
        self.pi.load_state_dict(d["pi"])
        self.pi_target.load_state_dict(d["pi_target"])

    def save_model(self):
        # define the save path, do not modify
        filepath = str(self.model_dir) + "/model_parameters_" + str(self.seed) + ".pt"

        torch.save(
            {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "pi": self.pi.state_dict(),
                "pi_target": self.pi_target.state_dict(),
            },
            filepath,
        )
        print("Saved model to", filepath, "...")


if __name__ == "__main__":
    agent = DDPGAgent()
    print("Initiating the agent")

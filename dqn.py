"""DQN: ANN for Q-approximation using Q-learning update with experience replay and target network for better training"""

import copy
import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DQN:
    def __init__(self, env, use_target_net=False, name=None):
        self.name = name
        if self.name is not None:
            self.tb = SummaryWriter("runs/" + name)

        self.env = env
        self.use_target_net = use_target_net
        self.q_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        if self.use_target_net:
            self.target_net = copy.deepcopy(self.q_net)
        self.loss = nn.MSELoss()
        self.buffer = deque(maxlen=100000)
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        self.discount = 0.99
        self.eps = 1
        self.eps_decay = 0.97
        self.min_eps = 0.01
        if self.name is not None:
            self.tb.add_graph(self.q_net, torch.zeros(env.observation_space.shape[0]))

    def act(self, state, explore=True):
        if explore and random.random() < self.eps:
            return self.env.action_space.sample()
        return torch.argmax(self.q_net(torch.from_numpy(state).float())).item()

    def replay(self, step=0, iterations=256, batch_size=64):
        for j in range(iterations):
            self.optimizer.zero_grad()
            X = torch.tensor(np.vstack(random.choices(self.buffer, k=batch_size))).float()
            s = self.env.observation_space.shape[0]
            tn = self.target_net if self.use_target_net else self.q_net
            target = X[:, s+1] + ((1 - X[:, -1]) * self.discount * torch.max(tn(X[:, -(s + 1):-1]), 1).values)
            loss = self.loss(torch.gather(self.q_net(X[:, 0:s]), 1, X[:, s].long().unsqueeze(-1)).flatten(), target)
            loss.backward()
            if self.name is not None:
                self.tb.add_scalar('training loss', loss, step + j)
            self.optimizer.step()

    def train(self, episodes=300, ep_len=1000, visualize=None, replay=1):
        rewards = []
        step = 0
        for i in tqdm(range(episodes)):
            reward = 0
            s = self.env.reset()
            d = False
            for j in range(ep_len):
                a = self.act(s)
                s_, r, d, _ = self.env.step(a)
                d_ = int(d and (j < self.env._max_episode_steps - 1))
                reward += r
                
                # Save to replay buffer
                self.buffer.append(np.hstack((s, a, r, s_, d_)))
                
                # Do replay
                if j % replay == 0:
                    self.replay(step, 1)
                    step += 1
                
                if d: break
                s = s_
            
            tqdm.write(f"Reward episode {i:03d}:{reward:12.6f}")
            if self.name is not None:
                self.tb.add_scalar('reward', reward, i)
            rewards.append(reward)
            
            if self.use_target_net and i % 5 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            self.replay(step)
            step += 256

            if self.eps > self.min_eps:
                self.eps *= self.eps_decay

            if visualize is not None and i % visualize == 0:
                s = self.env.reset()
                for i in range(ep_len):
                    self.env.render()
                    time.sleep(0.01)
                    s, _, d, _ = self.env.step(self.act(s, False))
                    if d: break
        
        return rewards


if __name__ == "__main__":
    # CartPole
    dqn = DQN(gym.make('CartPole-v0'), True, "dqn_cp_target5")
    dqn.train()
    dqn = DQN(gym.make('CartPole-v0'), False, "dqn_cp_notarget")
    dqn.train()

    # Lunar Lander
    dqn = DQN(gym.make('LunarLander-v2'), True, "dqn_ll_target5")
    dqn.train(1000)
    dqn = DQN(gym.make('LunarLander-v2'), False, "dqn_ll_notarget")
    dqn.train(1000)

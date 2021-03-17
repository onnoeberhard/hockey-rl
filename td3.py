"""TD3 implementation by Onno Eberhard. Based partially on pseudocode from [1], which is based on [2].

Additions: 
- Clipping actions at policy update step.
- Exploration noise decay
- Conservative target network updates [3], off by default

[1]: https://spinningup.openai.com/en/latest/algorithms/td3.html
[2]: https://arxiv.org/abs/1802.09477
[3]: https://openreview.net/forum?id=SJgn464tPB
"""

from collections import deque
from copy import deepcopy
import os
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# Reproducibility
random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

class TD3:
    def __init__(self, env, name=None, n_actions=None):
        self.name = name
        if self.name is not None:
            self.tb = SummaryWriter("runs/" + name)

        self.env_ = None
        if isinstance(env, str):
            self.make_env = lambda: gym.make(env)
        elif isinstance(env, tuple):
            self.make_env = env[0]
            self.env_ = tuple(e() for e in env)
        else:
            self.make_env = env
        self.env = self.make_env()

        # Only train actions 0...'n_acions' of action space
        if n_actions is not None:
            self.n_actions = n_actions
        else:
            self.n_actions = self.env.action_space.shape[0]

        # Hyperparameters
        self.buffer_size = 100_000
        self.hidden = 256             # Hidden units in neural network (3 layers + output)
        self.iterations = 10
        self.batch_size = 512
        self.update_frequency = 20    # Train every 'update_frequency' steps
        self.discount = 0.99
        self.polyak = 0.995
        self.noise_act = 0.2         # Exploration noise std
        self.noise_decay = 0.995
        self.noise_tps = 0.2          # Target policy smoothing noise std
        self.clip_noise = 0.5         # Clip tps noise to +-clip_noise
        self.policy_delay = 2         # How many Q-updates for one policy update
        self.ep_length = 1000
        self.explore_first = 10_000   # Follow random policy for first 'explore_first' steps for more exploration
        self.min_steps = 1000         # Minimum number of steps before first training
        self.conservative_target = False
        self.conservative_delta = 0.2
        self.target_eps = 10
        self.update_target_pol = 100   # How many policy updates for one possible target policy update

        # Initialize replay buffer, networks and optimizers
        self.buffer = deque(maxlen=self.buffer_size)
        self.policy = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_actions)
        )
        self.policy_target = deepcopy(self.policy)
        self.q1 = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0] + self.n_actions, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )
        self.q1_target = deepcopy(self.q1)
        self.q2 = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0] + self.n_actions, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )
        self.q2_target = deepcopy(self.q2)
        self.policy_optim = torch.optim.Adam(self.policy.parameters())
        self.q1_optim = torch.optim.Adam(self.q1.parameters())
        self.q2_optim = torch.optim.Adam(self.q2.parameters())


    def act(self, state, noise=False, target=False):
        with torch.no_grad():
            p = self.policy if not target else self.policy_target
            p.eval()
            a = p(torch.as_tensor(state).float())

            if target and noise:
                noise = self.noise_tps * torch.randn_like(a)

                # Clip noise
                noise = torch.min(
                    torch.cat((noise[None, ...], torch.tensor(self.clip_noise).expand_as(noise[None, ...]))), 0).values
                noise = torch.max(
                    torch.cat((noise[None, ...], torch.tensor(-self.clip_noise).expand_as(noise[None, ...]))), 0).values
                a += noise

            elif noise:
                a += self.noise_act * torch.randn_like(a)

            # Clip to valid action
            a = torch.min(torch.cat((a[None, ...],
                    torch.from_numpy(self.env.action_space.high[:self.n_actions]).expand_as(a[None, ...]))), 0).values
            a = torch.max(torch.cat((a[None, ...], 
                    torch.from_numpy(self.env.action_space.low[:self.n_actions]).expand_as(a[None, ...]))), 0).values

            # Sanity check: valid action
            assert (~a.isnan()).all()
            assert a.max() <= self.env.action_space.high.max()
            assert a.min() >= self.env.action_space.low.min()

            return a


    def train(self, episodes=5000, opponent=None, visualize=True, skip_episodes=0, add_reward=None, necessary_wins=0):
        self.best_policy = (-np.inf, self.policy)
        self.policy_updates = 0
        step = skip_episodes * self.ep_length
        streak = 0
        streak_done = False
        for i in tqdm(range(skip_episodes, episodes)):
            if self.env_:
                e = self.env_[i % len(self.env_)]
                if opponent:
                    op = opponent[i % len(self.env_)]
            else:
                e = self.env
                op = opponent
            if streak == necessary_wins:
                streak_done = True
            if i + 1 == episodes and not streak_done:
                i -= 1
            updates = 0
            reward = 0
            s = e.reset()
            for j in range(self.ep_length):
                # Do step and save training tuple in replay buffer
                a = (self.act(torch.from_numpy(s)[None, :], True).numpy()[0] 
                     if step - skip_episodes >= self.explore_first else e.action_space.sample()[:self.n_actions])
                if opponent:
                    a2 = op(e.obs_agent_two())
                    s_, r, d, info = e.step(np.hstack((a, a2)))
                else:
                    s_, r, d, info = e.step(a)
                if add_reward:
                    r += info[add_reward]
                d_ = int(d and ((j < e._max_episode_steps - 1) if isinstance(e, gym.wrappers.TimeLimit) else True))
                self.buffer.append(np.hstack((s, a, r, s_, d_)))
                s = s_
                reward += r
                step += 1
                if step == self.explore_first:
                    tqdm.write("Stopping following random policy!")

                # Do training if the time has come
                if step % self.update_frequency == 0 and step - skip_episodes > self.min_steps:
                    updates += 1
                    self.replay(step)

                if d: break

            tqdm.write(f"Reward episode {i:03d}:{reward:12.6f} ({updates} updates)")
            self.tb.add_scalar('Training Reward', reward, i)

            # Decay exploration noise
            self.noise_act *= self.noise_decay

            # Test current policy (every 10 episodes)
            if i % 10 == 0:
                won = 0
                rewards = np.zeros(5)
                for j in range(5):
                    if self.env_:
                        e = self.env_[j % len(self.env_)]
                    else:
                        e = self.env
                    s = e.reset()
                    w = 0
                    for k in range(self.ep_length):
                        if visualize:
                            e.render()
                        a = self.act(torch.from_numpy(s)[None, :]).numpy()[0]
                        if opponent:
                            a2 = op(e.obs_agent_two())
                            s, r, d, info = e.step(np.hstack((a, a2)))
                        else:
                            s, r, d, info = e.step(a)
                        rewards[j] += r
                        if 'winner' in info:
                            w += info['winner']
                        if d: break
                    tqdm.write(f"Winner: {w}")
                    won += w
                if won > 0:
                    streak += 1
                elif won < 0:
                    streak = 0

                # Policy checkpoint
                if not os.path.exists(f"./test_models/{self.name}/"):
                    os.mkdir(f"./test_models/{self.name}/")
                torch.save(self.policy.state_dict(), f"./test_models/{self.name}/p{i}.pt")

                if rewards.mean() > self.best_policy[0]:
                    self.best_policy = (rewards.mean(), self.policy)
                    torch.save(self.best_policy[1].state_dict(), f"./test_models/{self.name}/best_policy.pt")
                    tqdm.write("New best policy!")

                tqdm.write(f"Test reward mean:{rewards.mean():12.6f}")
                self.tb.add_scalar('Test Reward Mean', rewards.mean(), i)
                self.tb.add_scalar('Test Reward Std', rewards.std(), i)
                self.tb.add_histogram('policy_weights_0', self.policy[0].weight, i)
                self.tb.add_histogram('policy_bias_0', self.policy[0].bias, i)
                self.tb.add_histogram('policy_weights_2', self.policy[2].weight, i)
                self.tb.add_histogram('policy_bias_2', self.policy[2].bias, i)
                self.tb.add_histogram('policy_weights_4', self.policy[4].weight, i)
                self.tb.add_histogram('policy_bias_4', self.policy[4].bias, i)
                self.tb.add_histogram('q1_weights_0', self.q1[0].weight, i)
                self.tb.add_histogram('q1_bias_0', self.q1[0].bias, i)
                self.tb.add_histogram('q1_weights_2', self.q1[2].weight, i)
                self.tb.add_histogram('q1_bias_2', self.q1[2].bias, i)
                self.tb.add_histogram('q1_weights_4', self.q1[4].weight, i)
                self.tb.add_histogram('q1_bias_4', self.q1[4].bias, i)

            # Make checkpoint (every 100 episodes)
            if i + 1 == episodes:
                if not os.path.exists(f"./test_models/{self.name}/"):
                    os.mkdir(f"./test_models/{self.name}/")
                torch.save(self.policy.state_dict(), f"./test_models/{self.name}/policy.pt")
                torch.save(self.policy_target.state_dict(), f"./test_models/{self.name}/target_policy.pt")
                torch.save(self.q1.state_dict(), f"./test_models/{self.name}/q1.pt")
                torch.save(self.q1_target.state_dict(), f"./test_models/{self.name}/target_q1.pt")
                torch.save(self.q2.state_dict(), f"./test_models/{self.name}/q2.pt")
                torch.save(self.q2_target.state_dict(), f"./test_models/{self.name}/target_q2.pt")
            elif (i + 1) % 100 == 0:
                if not os.path.exists(f"./test_models/{self.name}/"):
                    os.mkdir(f"./test_models/{self.name}/")
                torch.save(self.policy.state_dict(), f"./test_models/{self.name}/policy_{i + 1}.pt")
                torch.save(self.policy_target.state_dict(), f"./test_models/{self.name}/target_policy_{i + 1}.pt")
                torch.save(self.q1.state_dict(), f"./test_models/{self.name}/q1_{i + 1}.pt")
                torch.save(self.q1_target.state_dict(), f"./test_models/{self.name}/target_q1_{i + 1}.pt")
                torch.save(self.q2.state_dict(), f"./test_models/{self.name}/q2_{i + 1}.pt")
                torch.save(self.q2_target.state_dict(), f"./test_models/{self.name}/target_q2_{i + 1}.pt")
            
        if self.env_:
            for e in self.env_:
                e.close()
        else:
            self.env.close()


    def replay(self, step=0):
        for i in range(self.iterations):
            # Draw minibatch and separate tuple
            X = torch.tensor(np.vstack(random.choices(self.buffer, k=self.batch_size))).float()
            sl = self.env.observation_space.shape[0]
            al = self.n_actions
            s = X[:, :sl]
            a = X[:, sl:(sl + al)]
            r = X[:, (sl + al):(sl + al + 1)]
            s_ = X[:, (sl + al + 1):-1]
            d = X[:, -1][:, None]

            # - Update Q functions -
            # Activate training mode and zero gradients
            self.q1.train()
            self.q2.train()
            self.q1_optim.zero_grad()
            self.q2_optim.zero_grad()

            # Forward pass
            with torch.no_grad():
                sa_ = torch.cat((s_, self.act(s_, True, True)), 1)
                y = r + (self.discount * (1 - d)
                         * torch.min(torch.cat((self.q1_target(sa_)[None, ...], 
                                                self.q2_target(sa_)[None, ...])), 0).values)

            q1_loss = F.mse_loss(self.q1(torch.cat((s, a), 1)), y)
            q2_loss = F.mse_loss(self.q2(torch.cat((s, a), 1)), y)
            
            self.tb.add_scalar('Q1 Loss', q1_loss, step + i)
            self.tb.add_scalar('Q2 Loss', q2_loss, step + i)

            # Backward pass and update step
            q1_loss.backward()
            q2_loss.backward()
            self.q1_optim.step()
            self.q2_optim.step()

            # - Update policy and target networks -
            if i % self.policy_delay == 0:
                self.policy_updates += 1
                # Update target Q networks
                for m, t in zip((self.q1, self.q2), (self.q1_target, self.q2_target)):
                    for mp, tp in zip(m.parameters(), t.parameters()):
                        tp.data = self.polyak*tp.data + (1 - self.polyak)*mp.data

                # Update policy
                self.policy.train()
                self.q1.eval()
                self.policy_optim.zero_grad()

                a = self.policy(s)

                # Clip to valid action (not done here in official algorithm!)
                a = torch.min(torch.cat((a[None, ...], torch.from_numpy(
                        self.env.action_space.high[:self.n_actions]).expand_as(a[None, ...]))), 0).values
                a = torch.max(torch.cat((a[None, ...], torch.from_numpy(
                        self.env.action_space.low[:self.n_actions]).expand_as(a[None, ...]))), 0).values

                loss = -self.q1(torch.cat((s, a), 1)).mean()
                loss.backward()
                self.policy_optim.step()

                self.tb.add_scalar('Policy Loss', loss, step + i)

                # - Update target policy -
                if not self.conservative_target:
                    for mp, tp in zip(self.policy.parameters(), self.policy_target.parameters()):
                        tp.data = self.polyak*tp.data + (1 - self.polyak)*mp.data

                # Conservative Policy Gradients method
                elif self.policy_updates >= self.update_target_pol:
                    self.policy_updates = 0
                    env = self.make_env()    # Don't disturb ongoing training with this
                    online_won = []
                    for j in range(self.target_eps):
                        # Simulate episode with target policy and online policy
                        reward_online = 0
                        reward_target = 0
                        s = env.reset()
                        for k in range(self.ep_length):
                            a = self.act(torch.from_numpy(s)[None, :], target=False).numpy()[0]
                            s, r, d, _ = env.step(a)
                            reward_online += r
                            if d: break
                        s = env.reset()
                        for k in range(self.ep_length):
                            a = self.act(torch.from_numpy(s)[None, :], target=True).numpy()[0]
                            s, r, d, _ = env.step(a)
                            reward_target += r
                            if d: break
                        
                        online_won.append(int(reward_online > reward_target))
                    
                    # Update target policy if online policy is better
                    tqdm.write(f"Target policy ratio: {np.mean(online_won):.1f}")
                    self.tb.add_scalar('Target policy ratio', np.mean(online_won), step + i)
                    if np.mean(online_won) >= 1 - self.conservative_delta:
                        tqdm.write("Target policy update!")
                        self.policy_target.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    name = "td3_h"
    # tb = SummaryWriter("runs/" + name)

    # env_name = 'LunarLanderContinuous-v2'
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    # View random agent
    env.reset()
    reward = 0
    for i in range(300):
        env.render()
        _, r, d, _ = env.step(env.action_space.sample())
        reward += r
        if d: break
    env.close()

    print(f"Reward random agent: {reward}")

    agent = TD3(env_name, name)
    agent.train()

    # Testing (300 episodes)
    rewards = np.full((300, 1000), np.nan)
    for i in tqdm(range(300)):
        s = env.reset()
        for j in range(1000):
            s, r, d, _ = env.step(agent.act(torch.from_numpy(s)[None, :]).numpy()[0])
            rewards[i, j] = r
            if d: break
        agent.tb.add_scalar("Final Test Rewards", np.nansum(rewards[i]), i)

    np.savetxt(name + "_test.csv", rewards, delimiter=',')
    torch.save(agent.policy.state_dict(), name + ".model")

    while True:
        s = env.reset()
        for j in range(1000):
            env.render()
            s, _, d, _ = env.step(agent.act(torch.from_numpy(s)[None, :]).numpy()[0])
            if d: break

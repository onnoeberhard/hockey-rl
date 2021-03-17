from copy import deepcopy
from os import name
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from hockey_env import BasicOpponent, HockeyEnv
from td3 import TD3
from itertools import count

skip = 0
skip_episodes = 0

agent = None
policy = policy_target = q1 = q1_target = q2 = q2_target = \
    policy_ = policy_target_ = q1_ = q1_target_ = q2_ = q2_target_ = None

if skip != 0:
    policy, policy_target, q1, q1_target, q2, q2_target = \
        [torch.load(f"test_models/hockey_step_{skip if skip_episodes == 0 else skip + 1}/"
                    f"{x}{'' if skip_episodes == 0 else f'_{skip_episodes}'}.pt") for x in 
            ("policy", "target_policy", "q1", "target_q1", "q2", "target_q2")]
if skip == 2:
    policy_, policy_target_, q1_, q1_target_, q2_, q2_target_ = \
        [torch.load(f"test_models/hockey_step_{skip - 1}/{x}.pt") for x in 
            ("policy", "target_policy", "q1", "target_q1", "q2", "target_q2")]
    

# First step: training camp (20_000 episodes) + basic opponent (10_000 episodes) + strong opponent (10_000 episodes)
strong_opponent = BasicOpponent(weak=False)
if skip < 5:
    weak_opponent = BasicOpponent()

    curriculum = [
            (lambda: HockeyEnv(mode=HockeyEnv.TRAIN_SHOOTING), lambda _: [0, 0, 0, 0], 5000),
            (lambda: HockeyEnv(mode=HockeyEnv.TRAIN_DEFENSE), lambda _: [0, 0, 0, 0], 5000),
            ((lambda: HockeyEnv(mode=HockeyEnv.TRAIN_SHOOTING), lambda: HockeyEnv(mode=HockeyEnv.TRAIN_DEFENSE),     
                lambda: HockeyEnv()), (lambda _: [0, 0, 0, 0], lambda _: [0, 0, 0, 0], weak_opponent.act), 10_000),
            (lambda: HockeyEnv(), weak_opponent.act, 10_000),
            (lambda: HockeyEnv(), strong_opponent.act, 5000)
    ]

    for i, (env, opponent, episodes) in enumerate(curriculum[skip:]):
        agent = TD3(env, f"hockey_step_{i + skip + 1}_fresh", 4)    # TODO

        # Reuse prior experience
        if policy is not None and policy_ is None:
            for m, s in zip((agent.policy, agent.policy_target, agent.q1, agent.q1_target, agent.q2, agent.q2_target), 
                            (policy, policy_target, q1, q1_target, q2, q2_target)):
                m.load_state_dict(s)
        elif policy is not None:
            for m, s1, s2 in zip(
                    (agent.policy, agent.policy_target, agent.q1, agent.q1_target, agent.q2, agent.q2_target), 
                    (policy, policy_target, q1, q1_target, q2, q2_target),
                    (policy_, policy_target_, q1, q1_target_, q2_, q2_target_)):
                s = deepcopy(s1)
                for w in s1:
                    s[w] = 0.5 * (s1[w] + s2[w])
                m.load_state_dict(s)
                policy_ = None

        add_reward = None
        if i >= 3:
            add_reward = 'reward_touch_puck'
            agent.noise_act = 0.5
            agent.noise_decay = 0.999
            agent.explore_first = 0

        agent.train(episodes, opponent, visualize=False, skip_episodes=skip_episodes, add_reward=add_reward)

        # Update experience
        for m, s in zip((agent.policy, agent.policy_target, agent.q1, agent.q1_target, agent.q2, agent.q2_target), 
                        (policy, policy_target, q1, q1_target, q2, q2_target)):
            s = m.state_dict()

if agent:
    print(agent.best_policy[1].state_dict())
    torch.save(agent.best_policy[1].state_dict(), "best_policy.pt")

# Second step: self-play + strong opponent, each for 1000 eps + 10 (averaged) wins in a row -> repeat.
best_policy = torch.load(f"best_policy.pt")
selfplayer = TD3(lambda: HockeyEnv(), n_actions=4)
selfplayer.policy.load_state_dict(best_policy)
opponent = None
agent = TD3((lambda: HockeyEnv(), lambda: HockeyEnv()), f"hockey_step_6", 4)

# Reuse prior experience
if policy is not None:
    for m, s in zip((agent.policy, agent.policy_target, agent.q1, agent.q1_target, agent.q2, agent.q2_target), 
                    (policy, policy_target, q1, q1_target, q2, q2_target)):
        m.load_state_dict(s)
for i in count():
    if opponent is None or isinstance(opponent, BasicOpponent):
        opponent = TD3(HockeyEnv())
        opponent.policy.load_state_dict(best_policy)
        agent.train(1000, opponent, visualize=False) # , necessary_wins=10)

    elif isinstance(opponent, TD3):
        opponent = BasicOpponent(weak=False)
        agent.train(1000, opponent, visualize=False)
    
    agent.train(episodes, opponent, visualize=False, skip_episodes=skip_episodes)

    best_policy = agent.best_policy[1].state_dict()
    selfplayer.policy.load_state_dict(best_policy)

    torch.save(best_policy, f"best_policy_6_fresh.pt")

    Update experience
    for m, s in zip((agent.policy, agent.policy_target, agent.q1, agent.q1_target, agent.q2, agent.q2_target), 
                    (policy, policy_target, q1, q1_target, q2, q2_target)):
        s = m.state_dict()


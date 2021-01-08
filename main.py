import gym
import random
import torch
import numpy as np
import argparse
import json
import time
from datetime import datetime
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter

from munchausen_agent import MDQNAgent


def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    config["seed"] = args.seed
    env = gym.make('LunarLander-v2')
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    env.seed(config['seed'])
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = MDQNAgent(state_size=200, action_size=4, config=config)
    agent.train_agent()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default="dqn", type=str)
    parser.add_argument('--buffer_size', default=1e5, type=int)
    parser.add_argument('--max_episode_steps', default=1000, type=int) 
    parser.add_argument('--locexp', default="", type=str) 
    arg = parser.parse_args()
    main(arg)


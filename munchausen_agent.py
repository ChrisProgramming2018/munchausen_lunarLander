import numpy as np
import random
import gym
import sys
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from gym import wrappers
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from datetime import datetime
from model import QNetwork



class MDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.seed = config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        self.env = gym.make(config["env_name"])
        self.env.seed(self.seed)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        self.env.action_space.seed(self.seed)
        self.state_size = state_size
        self.action_size = action_size
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.eps_start = 1.0
        self.eval = config["eval"]
        self.eps = config["eps"]
        self.min_eps = config["eps_end"]
        self.eps_frames = int(config["eps_frames"])
        self.frames = int(config["total_frames"])
        print("seed", self.seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer((state_size, ), (1, ), config["buffer_size"], self.seed, self.device)
        pathname = str(config["seed"]) + str(dt_string)
        tensorboard_name = str(config["locexp"]) + '/runs/' + "MDQN" + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.vid_path = str(config["locexp"])
        self.steps = 0
        self.entropy = 0.03
        self.alpha = 0.9
        self.clip = -1
    

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # not gradients on target
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach()
            q_next_diff = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(1)
            logsum = torch.logsumexp(q_next_diff/self.entropy, dim=1).unsqueeze(-1)
            tau_log_action_next = q_next_diff - self.entropy * logsum

            # compute policy
            target_policy = F.softmax(Q_targets_next/self.entropy, dim=1)

            # Compute Q targets for current states 
            Q_target =  (self.gamma * (target_policy * (Q_targets_next - tau_log_action_next) * dones).sum(dim=1)).unsqueeze(1)    # shape batch_size, 1
            # munchhaus add use target network
            m_target = self.qnetwork_target(states)
            m_target_max = m_target.max(1)[0]
            q_diff = m_target - m_target_max.unsqueeze(1)
            m_logsum = logsum = torch.logsumexp(q_diff/self.entropy, dim=1).unsqueeze(-1)
            log_pi = q_diff - self.entropy * m_logsum
            munchhaus_reward = log_pi.gather(1, actions).clamp(min=self.clip, max=0)
            Q_targets_munchausen = (rewards + self.alpha * munchhaus_reward) + Q_target 

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets_munchausen.detach())
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('loss', loss, self.steps)
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def train_agent(self):
        scores_window = deque(maxlen=100)
        eps = self.eps
        t0 = time.time()
        done = True
        episode_reward = 0
        i_epiosde = 1
        t = 0
        for frame in range(1, self.frames + 1):
            self.steps +=1
            t += 1
            if done:
                if i_epiosde % self.eval == 0:
                    self.eval_policy()
                i_epiosde += 1
                scores_window.append(episode_reward)
                ave_reward = np.mean(scores_window)
                print("Totalsteps {} Epiosde {} Steps {} Reward {:.2f} Reward averge{:.2f} eps {:.2f} Time {}".format(frame, i_epiosde, t, episode_reward, np.mean(scores_window), eps, time_format(time.time() - t0)))
                self.writer.add_scalar('Aver_reward', ave_reward, self.steps)
                self.writer.add_scalar('Reward', episode_reward, self.steps)
                t = 0
                episode_reward = 0
                state = self.env.reset()
            
            action = self.act(state, eps)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.memory.add(state, action, reward, next_state, done, done)
            self.learn()
            state = next_state
            if frame < self.eps_frames:
                eps = max(self.eps_start - (frame*(1/self.eps_frames)), self.min_eps)
            else:
                eps = max(self.min_eps - self.min_eps*((frame - self.eps_frames)/(self.frames- self.eps_frames)), 0.001)


    def eval_policy(self, eval_episodes=4):
        env = wrappers.Monitor(self.env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True, force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        s = 0
        for i_epiosde in range(eval_episodes):
            episode_reward = 0
            state = env.reset()
            while True:
                s += 1
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    scores_window.append(episode_reward)
                    break
        average_reward = np.mean(scores_window)
        print("Eval reward {}".format(average_reward))
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)

def time_format(sec):
    """
    Args:
    param1
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)


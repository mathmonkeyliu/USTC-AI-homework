import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    """A fixed-size replay buffer."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a mini-batch."""
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def size(self):
        """Return the current number of stored transitions."""
        return len(self.buffer)


class Qnet(nn.Module):
    """Small MLP used to approximate Q-values."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """Deep Q-Network agent."""
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        """Choose an action with epsilon-greedy exploration."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q_net(state).argmax(dim=1).item()

    def update(self, transition_dict):
        """Run one DQN update step."""
        states = torch.as_tensor(transition_dict["states"], device=self.device)
        actions = torch.as_tensor(transition_dict["actions"], device=self.device).view(-1, 1)
        rewards = torch.as_tensor(transition_dict["rewards"], device=self.device).view(-1, 1)
        next_states = torch.as_tensor(transition_dict["next_states"], device=self.device)
        dones = torch.as_tensor(transition_dict["dones"], device=self.device).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return loss.item()
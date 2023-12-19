import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, n_neurons, activations):
        super(QNetwork, self).__init__()
        layers = []
        for i in range(len(n_neurons) + 1):
            if i == 0:
                layers.append(nn.Linear(input_size, n_neurons[i]))
            elif i == len(n_neurons):
                layers.append(nn.Linear(n_neurons[i - 1], output_size))
            else:
                layers.append(nn.Linear(n_neurons[i - 1], n_neurons[i]))
            if i < len(n_neurons):
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TetrisQLAgent:
    def __init__(self, state_size, num_actions, mem_size, discount, epsilon_stop_episode, n_neurons, activations,
                 replay_start_size, seed, h5_checkpoint, deque_checkpoint):
        self.state_size = state_size
        self.num_actions = num_actions
        self.mem_size = mem_size
        self.discount = discount
        self.epsilon_stop_episode = epsilon_stop_episode
        self.replay_start_size = replay_start_size
        self.seed = seed
        self.h5_checkpoint = h5_checkpoint
        self.deque_checkpoint = deque_checkpoint

        # Initialize Q-networks
        self.q_network = QNetwork(state_size, num_actions, n_neurons, activations).cuda()
        self.target_q_network = QNetwork(state_size, num_actions, n_neurons, activations).cuda()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters())

        # Initialize replay memory
        self.replay_memory = []

    def best_state(self, next_states):
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).cuda()
        q_values = self.q_network(next_states_tensor)

        
        best_q, idx = torch.max(q_values, dim=0)
        return next_states[idx][0].tolist(), idx.item(), best_q.item()



    def add_to_memory(self, state, next_state, reward, done):
        self.replay_memory.append((state, next_state, reward, done))

    def train(self, save, h5_file, pkl_file, batch_size, epochs):
        if len(self.replay_memory) < self.replay_start_size:
            return

        batch = random.sample(self.replay_memory, batch_size)
        state_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)
        max_next_q_values, _ = torch.max(next_q_values, dim=1)
        target_q_values = reward_batch + (1 - done_batch) * self.discount * max_next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update_target_network()

    def soft_update_target_network(self):
        tau = 0.001  # Soft update parameter
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)
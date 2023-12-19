import keras
import tensorflow as tf
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import torch
# import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle


# import torchvision.transforms as transforms
# import torchvision.datasets as datasets


class MLP(nn.Module):
    def __init__(self, state_size, n_neuron_1, n_neuron_2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, n_neuron_1, bias=True)
        self.fc2 = nn.Linear(n_neuron_1, n_neuron_2, bias=True)
        self.fc3 = nn.Linear(n_neuron_2, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DQNAgent:
    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None, seed=10000,
                 h5_checkpoint=None, deque_checkpoint=None):

        assert len(activations) == len(n_neurons) + 1

        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.state_size = state_size
        if not deque_checkpoint:
            self.memory = deque(maxlen=mem_size)
        else:
            print("here")
            with open(deque_checkpoint, 'rb') as file:
                self.memory = pickle.load(file)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = activations
        self.criterion = nn.MSELoss()
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        if not h5_checkpoint:
            self.pytorch_model, self.keras_model, self.keras_weights = self._build_model()
        else:
            self.keras_model = load_model(h5_checkpoint)
            self.keras_weights = self.keras_model.get_weights()
            self.pytorch_model = MLP(state_size=self.state_size, n_neuron_1=self.n_neurons[0], n_neuron_2=self.n_neurons[1])
            self.pytorch_model.state_dict()['fc1.weight'].copy_(torch.from_numpy(self.keras_weights[0]).transpose(0, 1))
            self.pytorch_model.state_dict()['fc1.bias'].copy_(torch.from_numpy(self.keras_weights[1]))
            self.pytorch_model.state_dict()['fc2.weight'].copy_(torch.from_numpy(self.keras_weights[2]).transpose(0, 1))
            self.pytorch_model.state_dict()['fc2.bias'].copy_(torch.from_numpy(self.keras_weights[3]))
            self.pytorch_model.state_dict()['fc3.weight'].copy_(torch.from_numpy(self.keras_weights[4]).transpose(0, 1))
            self.pytorch_model.state_dict()['fc3.bias'].copy_(torch.from_numpy(self.keras_weights[5]))
            if torch.cuda.is_available():
                self.pytorch_model.cuda()
                # model = torch.nn.DataParallel(model)
                cudnn.benchmark = True

        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        keras_model = Sequential()
        keras_model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))
        for i in range(1, len(self.n_neurons)):
            keras_model.add(Dense(self.n_neurons[i], activation=self.activations[i]))
        keras_model.add(Dense(1, activation=self.activations[-1]))
        keras_model.compile(loss='mse', optimizer='adam')
        keras_weights = keras_model.get_weights()

        pytorch_model = MLP(state_size=self.state_size, n_neuron_1=self.n_neurons[0], n_neuron_2=self.n_neurons[1])
        pytorch_model.state_dict()['fc1.weight'].copy_(torch.from_numpy(keras_weights[0]).transpose(0, 1))
        pytorch_model.state_dict()['fc1.bias'].copy_(torch.from_numpy(keras_weights[1]))
        pytorch_model.state_dict()['fc2.weight'].copy_(torch.from_numpy(keras_weights[2]).transpose(0, 1))
        pytorch_model.state_dict()['fc2.bias'].copy_(torch.from_numpy(keras_weights[3]))
        pytorch_model.state_dict()['fc3.weight'].copy_(torch.from_numpy(keras_weights[4]).transpose(0, 1))
        pytorch_model.state_dict()['fc3.bias'].copy_(torch.from_numpy(keras_weights[5]))

        if torch.cuda.is_available():
            pytorch_model.cuda()
            # model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        return pytorch_model, keras_model, keras_weights

    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))

    def random_value(self):
        '''Random score for a certain action'''
        return random.random()

    def predict_value(self, states):
        '''Predicts the score for a certain state'''
        return self.pytorch_model(states)

    # def act(self, state):
    #     '''Returns the expected score of a certain state'''
    #     state = np.reshape(state, [1, self.state_size])
    #     if random.random() <= self.epsilon:
    #         return self.random_value()
    #     else:
    #         return self.predict_value(state)

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            idx = random.randint(0, states.shape[0] - 1)
            with torch.no_grad():
                return states[idx, :], idx, self.pytorch_model(states[idx, :])
        else:
            with torch.no_grad():
                predicts = self.pytorch_model(states)
            max_value, max_idx = torch.max(predicts.data, 0)

            best_state = states[max_idx, :]
            best_state = best_state.squeeze(0)
            # best_q = predicts[max_idx].cpu().item()

        return best_state, max_idx.item(), max_value

    def train(self, save, h5_file, pkl_file, batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:
            batch = random.sample(self.memory, batch_size)
            # self.model.train()
            # for epoch in range(epochs):
            #     # current_states = torch.cat([x[0] for x in batch], dim=0).cuda()
            #     current_states = torch.stack([x[0] for x in batch]).cuda().requires_grad_(True)
            #     next_qs = torch.tensor([x[1] for x in batch]).cuda()
            #     new_qs = torch.tensor([x[2] for x in batch]).type(torch.FloatTensor).cuda()
            #     dones = torch.tensor([x[3] for x in batch]).cuda() == False
            #
            #     new_qs[dones] = new_qs[dones] + self.discount * next_qs[dones]
            #     new_qs = new_qs.unsqueeze(1)
            #
            #     current_states, new_qs = map(Variable, (current_states, new_qs))
            #     outputs = self.model(current_states)
            #     loss = self.criterion(outputs, new_qs)
            #
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            #

            # Get the expected score for the next states, in batch (better performance)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.keras_model.predict(next_states)]

            x = []
            y = []
            rewards = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)
                rewards.append(reward)

            # Fit the model to the given values
            self.keras_model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            if save:
                save_model(self.keras_model, h5_file)
                with open(pkl_file, 'wb') as file:
                    pickle.dump(self.memory, file)

            self.keras_weights = self.keras_model.get_weights()

            self.pytorch_model.state_dict()['fc1.weight']. \
                copy_(torch.from_numpy(self.keras_weights[0]).transpose(0, 1).cuda())
            self.pytorch_model.state_dict()['fc1.bias']. \
                copy_(torch.from_numpy(self.keras_weights[1]).cuda())
            self.pytorch_model.state_dict()['fc2.weight']. \
                copy_(torch.from_numpy(self.keras_weights[2]).transpose(0, 1).cuda())
            self.pytorch_model.state_dict()['fc2.bias']. \
                copy_(torch.from_numpy(self.keras_weights[3])).cuda()
            self.pytorch_model.state_dict()['fc3.weight']. \
                copy_(torch.from_numpy(self.keras_weights[4]).transpose(0, 1).cuda())
            self.pytorch_model.state_dict()['fc3.bias']. \
                copy_(torch.from_numpy(self.keras_weights[5]).cuda())

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay


    def save_best(self, h5_best_file, pkl_best_file):
        save_model(self.keras_model, h5_best_file)
        with open(pkl_best_file, 'wb') as file:
            pickle.dump(self.memory, file)

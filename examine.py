from dqn_agent import DQNAgent
from tetris_player import Tetris
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle
import os
from tetris_generator import Tetris_Generator


max_steps = 2000
max_render_steps = 100
render_fps = 20
last_render_frame_delay = 5
moves_required = 5
lines_required = 3

generator = Tetris_Generator(L=lines_required, M=moves_required)
generator.reset()
initial_board = generator.board.astype(int).tolist()
sequence = generator.pieces


# initial_board = np.array(
#     [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
#     [1., 0., 1., 1., 1., 0., 0., 0., 1., 1.]])
# initial_board = initial_board.tolist()
# sequence = [6, 2, 5, 3, 1]


sequence.append(-1)
sequence.append(-1)
env = Tetris(0)
agent = DQNAgent(env.get_state_size(), n_neurons=[32, 32])
agent.epsilon = -1

log_dir = 'examine'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

current_state = torch.tensor(env.reset(moves_required, lines_required, initial_board, sequence)).cuda()

done = False
steps = 0
render = True

h5_file = 'best.checkpoint.episode.5479.winrate.1.0.h5'
keras_model = load_model(h5_file)
keras_weights = keras_model.get_weights()

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

pytorch_model = MLP(state_size=env.get_state_size(), n_neuron_1=32, n_neuron_2=32)
pytorch_model.state_dict()['fc1.weight'].copy_(torch.from_numpy(keras_weights[0]).transpose(0, 1))
pytorch_model.state_dict()['fc1.bias'].copy_(torch.from_numpy(keras_weights[1]))
pytorch_model.state_dict()['fc2.weight'].copy_(torch.from_numpy(keras_weights[2]).transpose(0, 1))
pytorch_model.state_dict()['fc2.bias'].copy_(torch.from_numpy(keras_weights[3]))
pytorch_model.state_dict()['fc3.weight'].copy_(torch.from_numpy(keras_weights[4]).transpose(0, 1))
pytorch_model.state_dict()['fc3.bias'].copy_(torch.from_numpy(keras_weights[5]))
pytorch_model.cuda()
cudnn.benchmark = True

agent.pytorch_model = pytorch_model

while not done:
    next_states, next_actions, next_boards = env.get_next_states()

    best_state, idx, best_q = agent.best_state(next_states)
    best_action = next_actions[idx]
    best_board = next_boards[idx]

    reward, done = env.play(best_action[0], best_action[1], best_board,
                            episode='examine', steps=steps, max_render_steps=max_render_steps, fps=render_fps,
                            last_frame_delay=last_render_frame_delay, log_dir=log_dir, render=render, render_delay=None)
    steps += 1
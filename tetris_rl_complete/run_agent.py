import torch
from agent import TetrisQLAgent
from tetris_player import Tetris
from tetris_generator import Tetris_Generator
from datetime import datetime
from tqdm import tqdm
import os
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='CSI 5340 Project: Tetris')
parser.add_argument('--seed', default=10001, type=int, help='random seed')
parser.add_argument('--episodes', default=10000, type=int, help='number of episodes')
parser.add_argument('--eps-stop', default=1, type=int)
parser.add_argument('--mem', default=4000, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--replay', default=1000, type=int)
parser.add_argument('--M', default=5, type=int)
parser.add_argument('--L', default=3, type=int)
args = parser.parse_args()


def main():
    seed = args.seed
    env = Tetris(seed)
    moves_required = args.M
    lines_required = args.L

    episodes = args.episodes
    max_steps = 20000
    epsilon_stop_episode = args.eps_stop
    mem_size = args.mem
    discount = 0.95
    batch_size = args.batch_size
    epochs = args.epochs
    render_every = None
    log_every = 10
    replay_start_size = args.replay
    train_every = 10
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']
    render_fps = 20
    last_render_frame_delay = 5
    max_render_steps = 100

    agent = TetrisQLAgent(env.get_state_size(), num_actions=moves_required,
                          mem_size=mem_size, discount=discount,
                          epsilon_stop_episode=epsilon_stop_episode, n_neurons=n_neurons,
                          activations=activations, replay_start_size=replay_start_size, seed=seed)

    generator = Tetris_Generator(L=lines_required, M=moves_required)

    scores = []

    if log_every:
        logname = (f'log.csv')
        if not os.path.exists(logname):
            with open(logname, 'w', newline='') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['episode', 'winrate'])

    winrate_best = 0

    for episode in tqdm(range(episodes)):
        generator.reset()
        initial_board = generator.board.astype(int).tolist()
        sequence = generator.pieces
        sequence.append(-1)

        current_state = torch.tensor(env.reset(moves_required, lines_required, initial_board, sequence)).cuda()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        while not done and (not max_steps or steps < max_steps):
            next_states, next_actions, next_boards = env.get_next_states()
            best_state_idx = agent.best_action(next_states)
            best_state = next_states[best_state_idx, :]
            best_action = next_actions[best_state_idx]
            best_board = next_boards[best_state_idx]

            reward, done = env.play(best_action[0], best_action[1], best_board,
                                    episode, steps, max_render_steps, render_fps, last_render_frame_delay,
                                    render=render, render_delay=render_delay)

            if not done:
                agent.add_to_memory(current_state.tolist(), best_state.tolist(), reward, done)
            else:
                agent.add_to_memory(current_state.tolist(), best_state.tolist(), 0, False)
                agent.add_to_memory(best_state.tolist(), [0 for _ in range(env.get_state_size())], reward, done)

            current_state = best_state
            steps += 1

        scores.append(env.get_game_score())

        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        if log_every and episode and (episode + 1) % log_every == 0:
            record = np.array(scores[-log_every:])
            winrate = (record > 0).sum() / log_every

            if winrate > winrate_best:
                winrate_best = winrate

            with open(logname, 'a', newline='') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([episode, winrate])

    print('Best winrate: {}%'.format(winrate_best * 100))


if __name__ == "__main__":
    main()

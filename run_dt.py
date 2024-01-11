# import torch
# from dqn_agent import DQNAgent
# from tetris_player import Tetris
# from tetris_generator import Tetris_Generator
# from datetime import datetime
# from statistics import mean, median
# import random
# # from logs import CustomTensorBoard
# from tqdm import tqdm
# import os
# import csv
# import pickle
# import argparse
# import numpy as np
# import pygame



# parser = argparse.ArgumentParser(description='CSI 5340 Project: Tetris')
# parser.add_argument('--seed', default=10001, type=int, help='random seed')
# parser.add_argument('--episodes', default=10000, type=int, help='number of episodes')
# parser.add_argument('--eps-stop', default=1, type=int)
# parser.add_argument('--mem', default=4000, type=int)
# parser.add_argument('--batch-size', default=256, type=int)
# parser.add_argument('--epochs', default=3, type=int)
# parser.add_argument('--replay', default=1000, type=int)
# parser.add_argument('--M', default=10, type=int)
# parser.add_argument('--L', default=10, type=int)
# args = parser.parse_args()


# # Run dqn with Tetris
# def dqn():
#     seed = args.seed
#     env = Tetris()
    
#     initial_configurations_file = ''
#     episodes = args.episodes
#     max_steps = 20000
#     epsilon_stop_episode = args.eps_stop
#     mem_size = args.mem
#     discount = 0.95
#     batch_size = args.batch_size
#     epochs = args.epochs
#     render_every = 0
#     log_every = 10
#     replay_start_size = args.replay
#     train_every = 10
#     n_neurons = [32, 32]
#     render_delay = None
#     activations = ['relu', 'relu', 'linear']
#     render_fps = 20
#     last_render_frame_delay = 5
#     max_render_steps = 40
#     save_model_every = 50
#     save_best = True
#     h5_checkpoint = ''
#     deque_checkpoint = ''
#     current_level = 0
#     levels = [
#             {"lines_to_clear": 3, "moves_limit": 35},
#             {"lines_to_clear": 5, "moves_limit": 35},
#             {"lines_to_clear": 7, "moves_limit": 45},
#             {"lines_to_clear": 9, "moves_limit": 50},
#             {"lines_to_clear": 12, "moves_limit": 55},
#             {"lines_to_clear": 15, "moves_limit": 60},
#             {"lines_to_clear": 18, "moves_limit": 65},
#             {"lines_to_clear": 20, "moves_limit": 70},
#             {"lines_to_clear": 22, "moves_limit": 70},
#             {"lines_to_clear": 25, "moves_limit": 70},

#         ]
#     moves_required = levels[current_level]["moves_limit"]
#     lines_required = levels[current_level]["lines_to_clear"]
#     pygame.init()
#     screen_width, screen_height = 550, 700
#     screen = pygame.display.set_mode((screen_width, screen_height))
#     clock = pygame.time.Clock()
#     screen.fill((255, 255, 255)) 
#     pygame.display.set_caption("Tetris")

#     agent = DQNAgent(env.get_state_size(),
#                  n_neurons=n_neurons, activations=activations,
#                  epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
#                  discount=discount, replay_start_size=replay_start_size, seed=seed,
#                  h5_checkpoint=None, deque_checkpoint=None)

#     generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"])

#     # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
#     # log = CustomTensorBoard(log_dir=log_dir)
#     log_dir = 'logs'
#     if not os.path.isdir(log_dir):
#         os.mkdir(log_dir)
#     log_dir = os.path.join(log_dir, 'tetris.M.{}.L.{}.nn.{}.{}.mem.{}.replay.{}.bs.{}.e.{}.episodes.{}.epsilon.stop.{}.seed.{}.{}'.
#                            format(moves_required, lines_required, n_neurons[0], n_neurons[1], mem_size, replay_start_size,
#                                   batch_size, epochs, episodes, epsilon_stop_episode, seed,
#                                   datetime.now().strftime("%Y%m%d-%H%M%S")))
#     if not os.path.isdir(log_dir):
#         os.mkdir(log_dir)

#     if save_model_every:
#         checkpoint_dir = os.path.join(log_dir, 'checkpoints')
#         if not os.path.isdir(checkpoint_dir):
#             os.mkdir(checkpoint_dir)

#     # initial_configurations = np.load(initial_configurations_file)
#     # initial_boards = initial_configurations['array1']
#     # sequences = initial_configurations['array2']
#     # initial_configurations.close()

#     scores = []

#     if log_every:
#         logname = (log_dir + '/log.csv')
#         if not os.path.exists(logname):
#             with open(logname, 'w', newline='') as logfile:
#                 logwriter = csv.writer(logfile, delimiter=',')
#                 logwriter.writerow(['episode', 'winrate'])

#     score_best = 0
#     episode_best = 0
#     steps_best = 0
#     winrate_best = 0

#     for episode in tqdm(range(episodes)):
#         generator.reset()
#         initial_board = generator.board.astype(int).tolist()
#         sequence = generator.pieces
#         sequence.append(-1)

#         current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
#         done = False
#         steps = 0
#         h5_file = ''
#         pkl_file = ''

#         if render_every and episode % render_every == 0:
#             render = True
#         else:
#             render = False

#         # Game
#         while not done and (not max_steps or steps < max_steps):
#             screen.fill((255, 255, 255))
#             next_states, next_actions, next_boards = env.get_next_states()
#             best_state, idx, best_q = agent.best_state(next_states)
#             best_action = next_actions[idx]
#             best_board = next_boards[idx]

#             reward, done, level_complete = env.play(best_action[0], best_action[1], best_board,
#                                     episode, steps, max_render_steps, render_fps, last_render_frame_delay,
#                                     log_dir, screen, render=render, render_delay=render_delay)

#             # agent.add_to_memory(current_state, best_q, reward, done)
#             if not done:
#                 agent.add_to_memory(current_state.tolist(), best_state.tolist(), reward, done)
#             else:
#                 agent.add_to_memory(current_state.tolist(), best_state.tolist(), 0, False)
#                 agent.add_to_memory(best_state.tolist(), [0 for _ in range(env.get_state_size())], reward, done)
            
#             if level_complete:
#                 env = Tetris()
#                 current_level+=1
#                 generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"])

#                 initial_board = generator.board.astype(int).tolist()
#                 sequence = generator.pieces
#                 sequence.append(-1)

#                 current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
#             # if done:
#             #     current_state = torch.tensor(env.reset()).cuda()
#             #     done = False
#             # else:
#             #     current_state = best_state
#             current_state = best_state
#             steps += 1

#             episode += 1
#             steps += 1
#             clock.tick(render_fps)



#         scores.append(env.get_game_score())
#         # if scores[-1] > score_best:
#         #     score_best = scores[-1]
#         #     episode_best = episode
#         #     steps_best = steps
#         #     if save_best:
#         #         h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
#         #         pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
#         #         agent.save_best(h5_best_file, pkl_best_file)
#         # Train
#         if episode % train_every == 0:
#             if save_model_every and (episode + 1) % save_model_every == 0:
#                 save = True
#                 h5_file = os.path.join(checkpoint_dir, 'episode.{}.checkpoint.h5'.format(episode))
#                 pkl_file = os.path.join(checkpoint_dir, 'episode.{}.deque.pkl'.format(episode))
#             else:
#                 save = False
#             agent.train(save, h5_file, pkl_file, batch_size=batch_size, epochs=epochs)

#         # Logs
#         if log_every and episode and (episode + 1) % log_every == 0:
#             record = np.array(scores[-log_every:])
#             winrate = (record > 0).sum() / log_every
#             # if winrate > winrate_best:
#             # if winrate >= winrate_best:
#             if winrate > 0.85:
#                 winrate_best = winrate
#                 episode_best = episode
#                 if save_best:
#                     h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
#                     pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
#                     agent.save_best(h5_best_file, pkl_best_file)

#             with open(logname, 'a', newline='') as logfile:
#                 logwriter = csv.writer(logfile, delimiter=',')
#                 logwriter.writerow([episode, winrate])

#             # log.log(episode, avg_score=avg_score, min_score=min_score,
#             #         max_score=max_score)
#     print('Best episode: {}'.format(episode_best))
#     print('Best winrate: {}%'.format(winrate_best * 100))
#     pygame.quit()

#     if save_best:
#         h5_best_file_new = os.path.join(checkpoint_dir, 'best.checkpoint.episode.{}.winrate.{}.h5'.
#                                         format(episode_best, winrate_best))
#         pkl_best_file_new = os.path.join(checkpoint_dir, 'best.deque.episode.{}.winrate.{}.pkl'.
#                                          format(episode_best, winrate_best))
#         os.rename(h5_best_file, h5_best_file_new)
#         os.rename(pkl_best_file, pkl_best_file_new)

# if __name__ == "__main__":
#     dqn()

import torch
from dqn_agent import DQNAgent
from tetris_player import Tetris
from tetris_generator import Tetris_Generator
from datetime import datetime
from statistics import mean, median
import random
# from logs import CustomTensorBoard
from tqdm import tqdm
import os
import csv
import pickle
import argparse
import numpy as np
import pygame



parser = argparse.ArgumentParser(description='CSI 5340 Project: Tetris')
parser.add_argument('--seed', default=10001, type=int, help='random seed')
parser.add_argument('--episodes', default=1000, type=int, help='number of episodes')
parser.add_argument('--eps-stop', default=1, type=int)
parser.add_argument('--mem', default=4000, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--replay', default=1000, type=int)
parser.add_argument('--M', default=5, type=int)
parser.add_argument('--L', default=3, type=int)
args = parser.parse_args()


# Run dqn with Tetris
def dqn():
    pygame.init()
    screen_width, screen_height = 950, 1020
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    seed = args.seed
    env = Tetris()
   
    initial_configurations_file = ''
    episodes = args.episodes
    max_steps = 20000
    epsilon_stop_episode = args.eps_stop
    mem_size = args.mem
    discount = 0.95
    batch_size = args.batch_size
    epochs = args.epochs
    render_every = 2
    log_every = 10
    replay_start_size = args.replay
    train_every = 10
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']
    render_fps = 20
    last_render_frame_delay = 5
    max_render_steps = 40
    save_model_every = 50
    save_best = True
    h5_checkpoint = 'logs/tetris.M.35.L.3.nn.32.32.mem.4000.replay.1000.bs.256.e.3.episodes.1000.epsilon.stop.1.seed.10001.20231219-102516/checkpoints/best.checkpoint.episode.989.winrate.0.9.h5'
    deque_checkpoint = 'logs/tetris.M.35.L.3.nn.32.32.mem.4000.replay.1000.bs.256.e.3.episodes.1000.epsilon.stop.1.seed.10001.20231219-102516/checkpoints/best.deque.episode.989.winrate.0.9.pkl'
    current_level = 0
    levels = [
            {"lines_to_clear": 3, "moves_limit": 35},
            {"lines_to_clear": 5, "moves_limit": 35},
            {"lines_to_clear": 7, "moves_limit": 45},
            {"lines_to_clear": 9, "moves_limit": 50},
            {"lines_to_clear": 12, "moves_limit": 55},
            {"lines_to_clear": 15, "moves_limit": 60},
            {"lines_to_clear": 18, "moves_limit": 65},
            {"lines_to_clear": 20, "moves_limit": 70},
            {"lines_to_clear": 22, "moves_limit": 70},
            {"lines_to_clear": 25, "moves_limit": 70},

        ]
    moves_required = levels[current_level]["moves_limit"]
    lines_required = levels[current_level]["lines_to_clear"]

    # random.seed(seed)
    # keras.utils.set_random_seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, seed=seed,
                     h5_checkpoint=h5_checkpoint, deque_checkpoint=deque_checkpoint)

    generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"])

    # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)
    log_dir = 'logs'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, 'tetris.M.{}.L.{}.nn.{}.{}.mem.{}.replay.{}.bs.{}.e.{}.episodes.{}.epsilon.stop.{}.seed.{}.{}'.
                           format(moves_required, lines_required, n_neurons[0], n_neurons[1], mem_size, replay_start_size,
                                  batch_size, epochs, episodes, epsilon_stop_episode, seed,
                                  datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if save_model_every:
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

    # initial_configurations = np.load(initial_configurations_file)
    # initial_boards = initial_configurations['array1']
    # sequences = initial_configurations['array2']
    # initial_configurations.close()

    scores = []

    if log_every:
        logname = (log_dir + '/log.csv')
        if not os.path.exists(logname):
            with open(logname, 'w', newline='') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['episode', 'winrate'])

    score_best = 0
    episode_best = 0
    steps_best = 0
    winrate_best = 0

    for episode in tqdm(range(episodes)):
        generator.reset()
        initial_board = generator.board.astype(int).tolist()
        sequence = generator.pieces
        sequence.append(-1)
        

        current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
        done = False
        steps = 0
        h5_file = ''
        pkl_file = ''

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            screen.fill((255, 255, 255))
            next_states, next_actions, next_boards = env.get_next_states()
            best_state, idx, best_q = agent.best_state(next_states)
            best_action = next_actions[idx]
            best_board = next_boards[idx]

            reward, done, level_complete = env.play(best_action[0], best_action[1], best_board,
                                    episode, steps, max_render_steps, render_fps, last_render_frame_delay,
                                    log_dir, render=render, render_delay=render_delay, screen=screen)
            
            if level_complete and current_level < 9:
                
                scores.append(env.get_game_score())
               
                env = Tetris()
                current_level+=1
                generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"])

                initial_board = generator.board.astype(int).tolist()
                sequence = generator.pieces
                sequence.append(-1)
            

                current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
                ######################
                # Uncomment Training #
                ######################
                # if episode % train_every == 0:
                #     if save_model_every and (episode + 1) % save_model_every == 0:
                #         save = True
                #         h5_file = os.path.join(checkpoint_dir, 'episode.{}.checkpoint.h5'.format(episode))
                #         pkl_file = os.path.join(checkpoint_dir, 'episode.{}.deque.pkl'.format(episode))
                #     else:
                #         save = False
                #     agent.train(save, h5_file, pkl_file, batch_size=batch_size, epochs=epochs)

                # # Logs
                # if log_every and episode and (episode + 1) % log_every == 0:
                #     record = np.array(scores[-log_every:])
                #     winrate = (record > 0).sum() / log_every
                #     # if winrate > winrate_best:
                #     # if winrate >= winrate_best:
                #     if winrate > 0.85:
                #         winrate_best = winrate
                #         episode_best = episode
                #         if save_best:
                #             h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
                #             pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
                #             agent.save_best(h5_best_file, pkl_best_file)

                #     with open(logname, 'a', newline='') as logfile:
                #         logwriter = csv.writer(logfile, delimiter=',')
                #         logwriter.writerow([episode, winrate])
                    
                # print(f"Winrate",winrate)

            elif level_complete:
                scores.append(env.get_game_score())
                done = True
                current_level=0


                ######################
                # Uncomment Training #
                ######################
                # if episode % train_every == 0:
                #     if save_model_every and (episode + 1) % save_model_every == 0:
                #         save = True
                #         h5_file = os.path.join(checkpoint_dir, 'episode.{}.checkpoint.h5'.format(episode))
                #         pkl_file = os.path.join(checkpoint_dir, 'episode.{}.deque.pkl'.format(episode))
                #     else:
                #         save = False
                #     agent.train(save, h5_file, pkl_file, batch_size=batch_size, epochs=epochs)

                # # Logs
                # if log_every and episode and (episode + 1) % log_every == 0:
                #     record = np.array(scores[-log_every:])
                #     winrate = (record > 0).sum() / log_every
                #     # if winrate > winrate_best:
                #     # if winrate >= winrate_best:
                #     if winrate > 0.85:
                #         winrate_best = winrate
                #         episode_best = episode
                #         if save_best:
                #             h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
                #             pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
                #             agent.save_best(h5_best_file, pkl_best_file)

                #     with open(logname, 'a', newline='') as logfile:
                #         logwriter = csv.writer(logfile, delimiter=',')
                #         logwriter.writerow([episode, winrate])


            # agent.add_to_memory(current_state, best_q, reward, done)
            if not done:
                agent.add_to_memory(current_state.tolist(), best_state.tolist(), reward, done)
            else:
                agent.add_to_memory(current_state.tolist(), best_state.tolist(), 0, False)
                agent.add_to_memory(best_state.tolist(), [0 for _ in range(env.get_state_size())], reward, done)
            # if done:
            #     current_state = torch.tensor(env.reset()).cuda()
            #     done = False
            # else:
            #     current_state = best_state
            current_state = best_state
            steps += 1
            # clock.tick(render_fps)
    

        scores.append(env.get_game_score())
        # if scores[-1] > score_best:
        #     score_best = scores[-1]
        #     episode_best = episode
        #     steps_best = steps
        #     if save_best:
        #         h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
        #         pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
        #         agent.save_best(h5_best_file, pkl_best_file)
        # Train
        if episode % train_every == 0:
            if save_model_every and (episode + 1) % save_model_every == 0:
                save = True
                h5_file = os.path.join(checkpoint_dir, 'episode.{}.checkpoint.h5'.format(episode))
                pkl_file = os.path.join(checkpoint_dir, 'episode.{}.deque.pkl'.format(episode))
            else:
                save = False
            agent.train(save, h5_file, pkl_file, batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and (episode + 1) % log_every == 0:
            record = np.array(scores[-log_every:])
            winrate = (record > 0).sum() / log_every
            # if winrate > winrate_best:
            # if winrate >= winrate_best:
            if winrate > 0.85:
                winrate_best = winrate
                episode_best = episode
                if save_best:
                    h5_best_file = os.path.join(checkpoint_dir, 'best.checkpoint.h5')
                    pkl_best_file = os.path.join(checkpoint_dir, 'best.deque.pkl')
                    agent.save_best(h5_best_file, pkl_best_file)

            with open(logname, 'a', newline='') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([episode, winrate])

            # log.log(episode, avg_score=avg_score, min_score=min_score,
            #         max_score=max_score)
    print('Best episode: {}'.format(episode_best))
    print('Best winrate: {}%'.format(winrate_best * 100))
    pygame.quit()

    if save_best:
        h5_best_file_new = os.path.join(checkpoint_dir, 'best.checkpoint.episode.{}.winrate.{}.h5'.
                                        format(episode_best, winrate_best))
        pkl_best_file_new = os.path.join(checkpoint_dir, 'best.deque.episode.{}.winrate.{}.pkl'.
                                         format(episode_best, winrate_best))
        os.rename(h5_best_file, h5_best_file_new)
        os.rename(pkl_best_file, pkl_best_file_new)

if __name__ == "__main__":
    dqn()
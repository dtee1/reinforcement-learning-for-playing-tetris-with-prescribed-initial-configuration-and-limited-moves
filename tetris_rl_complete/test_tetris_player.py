from tetris_player import Tetris
from tetris_generator import Tetris_Generator
import torch
import pygame

def run_tetris():
    pygame.init()
    screen_width, screen_height = 550, 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Tetris")


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

    env = Tetris()
    generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"], level=current_level)

    initial_board = generator.board.astype(int).tolist()
    sequence = generator.pieces
    sequence.append(-1)

    current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
    done = False
    steps = 0
    episode = 0
    render_fps = 1
    last_frame_delay = 5

    run = True
    max_render_steps = 100
    fps = 20
    last_frame_delay = 0.0
    log_dir = None
    clock = pygame.time.Clock()

    while run and not done:
       

        screen.fill((255, 255, 255))
        next_states, next_actions, next_boards = env.get_next_states()
        best_state, idx, best_q = [next_states[0]], 0, 0  # Replace with your logic for selecting the best state
        best_action = next_actions[idx]
        best_board = next_boards[idx]

        reward, done, level_complete = env.play(best_action[0], best_action[1], best_board,
                                episode, steps, max_render_steps, fps, last_frame_delay, log_dir, screen, render=True,
                                render_delay=0.05)
        if level_complete:
            env = Tetris()
            current_level+=1
            generator = Tetris_Generator(L=levels[current_level]["lines_to_clear"], M=levels[current_level]["moves_limit"], level=current_level)

            initial_board = generator.board.astype(int).tolist()
            sequence = generator.pieces
            sequence.append(-1)

            current_state = torch.tensor(env.reset(generator.M, generator.L, initial_board, sequence, current_level)).cuda()
        if done:
            run = False

        episode += 1
        steps += 1
        clock.tick(render_fps)

    pygame.quit()

if __name__ == "__main__":
    run_tetris()

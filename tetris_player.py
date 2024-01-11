import random
import numpy as np
import torch
from PIL import Image
from time import sleep
import matplotlib.pyplot as plt
import os
import imageio
from collections import deque
import pygame
import time

# Tetris game class
class Tetris:
    '''Tetris game class'''

    # Constants for the game board
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    # Tetromino shapes and rotations
    """
        I Tetromino (Key: 0)

        Rotation 0:
        (0,0) (1,0) (2,0) (3,0)

        Rotation 90:
        (1,0)
        (1,1)
        (1,2)
        (1,3)

        Rotation 180:
        (3,0) (2,0) (1,0) (0,0)

        Rotation 270:
        (1,3)
        (1,2)
        (1,1)
        (1,0)

        T Tetromino (Key: 1)
        Rotation 0:
        (1,0)
        (0,1) (1,1) (2,1)

        Rotation 90:
        (0,1)
        (1,2)
        (1,1)
        (1,0)

        Rotation 180:
        (1,2)
        (2,1) (1,1) (0,1)

        Rotation 270:
        (2,1)
        (1,0)
        (1,1)
        (1,2)

        S Tetromino (Key: 5)
        Rotation 0:
                (2,0) (1,0)
        (0,1) (1,1)

        Rotation 90:
        (0,0)
        (0,1) (1,1)
                (1,2)

        Rotation 180:
        (0,1) (1,1)
                (1,0) (2,0)

        Rotation 270:
                (1,2)
        (1,1) (0,1)
        (0,0)

        O Tetromino (Key: 6)
        Rotation 0:
        (1,0) (2,0)
        (1,1) (2,1)

        Rotation 90:
        (1,0) (2,0)
        (1,1) (2,1)

        Rotation 180:
        (1,0) (2,0)
        (1,1) (2,1)

        Rotation 270:
        (1,0) (2,0)
        (1,1) (2,1)
    """
    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        },
        -1: {  # None
            0: []
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    PIECES_SHAPES = {
        None: np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        0: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        1: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        2: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        3: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        4: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        5: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,

        6: np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,
        -1: np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]) * MAP_PLAYER,
    }

    RENDER_LEFT_SPACE = 0.1
    RENDER_BOTTOM_SPACE = 0.1
    BOARD_RENDER_WIDTH = 0.4
    BOARD_RENDER_HEIGHT = 0.8
    PIECE_RENDER_WIDTH = 0.2
    PIECE_RENDER_HEIGHT = 0.2
    TEXT_RENDER_WIDTH = 0.2
    TEXT_RENDER_HEIGHT = 0.2

    RECT_BOARD = [RENDER_LEFT_SPACE, RENDER_BOTTOM_SPACE, BOARD_RENDER_WIDTH,
                  BOARD_RENDER_HEIGHT]  # Left half for the board array
    RECT_PIECE = [RENDER_LEFT_SPACE + BOARD_RENDER_WIDTH + 0.15,
                  RENDER_BOTTOM_SPACE + BOARD_RENDER_HEIGHT - PIECE_RENDER_HEIGHT, PIECE_RENDER_WIDTH,
                  PIECE_RENDER_HEIGHT]  # Top-right quadrant for the smaller array
    RECT_TEXT_1 = [RENDER_LEFT_SPACE + BOARD_RENDER_WIDTH + 0.11,
                   RENDER_BOTTOM_SPACE + BOARD_RENDER_HEIGHT - TEXT_RENDER_HEIGHT - 0.13, TEXT_RENDER_WIDTH,
                   TEXT_RENDER_HEIGHT]
    RECT_TEXT_2 = [RENDER_LEFT_SPACE + BOARD_RENDER_WIDTH + 0.11,
                   RENDER_BOTTOM_SPACE + BOARD_RENDER_HEIGHT - TEXT_RENDER_HEIGHT - 0.18, TEXT_RENDER_WIDTH,
                   TEXT_RENDER_HEIGHT]
    RECT_TEXT_3 = [RENDER_LEFT_SPACE + BOARD_RENDER_WIDTH + 0.15,
                   RENDER_BOTTOM_SPACE + BOARD_RENDER_HEIGHT - TEXT_RENDER_HEIGHT - 0.31, TEXT_RENDER_WIDTH,
                   TEXT_RENDER_HEIGHT]
    RECT_TEXT_4 = [RENDER_LEFT_SPACE + BOARD_RENDER_WIDTH + 0.15,
                   RENDER_BOTTOM_SPACE + BOARD_RENDER_HEIGHT - TEXT_RENDER_HEIGHT - 0.36, TEXT_RENDER_WIDTH,
                   TEXT_RENDER_HEIGHT]

    GAMEOVER_TAG_1 = {
        0: 'mission passed!',
        1: 'mission failed',
        2: 'wasted'
    }
    GAMEOVER_TAG_2 = {
        0: 'RESPECT +',
        1: None,
        2: None
    }

    def __init__(self):
     
        self.block_size = 50
        self.level_complete = False

    def reset(self, moves_required, lines_required, initial_board, sequence, level=1):
        '''Resets the game, returning the current state'''
        self.moves_required = moves_required
        self.lines_required = lines_required
        self.moves_left = moves_required
        self.lines_left = lines_required
        self.board = initial_board
        self.game_over = False
        self.sequence = deque(sequence)
        self.level = level

        self.next_piece = self.sequence.popleft()
        self._new_round()
        self.score = 0
        self.frames = []
        self.images_saved = []
        return self._get_initial_board_props(self.board)

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        if self.current_piece >= 0:
            return Tetris.TETROMINOS[self.current_piece][self.current_rotation]
        else:
            return Tetris.TETROMINOS[-1][0]

    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score

    def _new_round(self):
        '''Starts a new round (new piece)'''
        self.current_piece = self.next_piece
        if self.current_piece >= 0:
            self.next_piece = self.sequence.popleft()
            self.current_pos = [3, 0]
            self.current_rotation = 0

            if self._check_collision(self._get_rotated_piece(), self.current_pos):
                self.game_over = True

    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i + 1:] if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)

        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i + 1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i + 1])

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    # def piece_onehot(self, piece_id):
    #
    def _get_horizontal_cell_width(self):
        '''Calculate the horizontal cell width of the current block at its current rotation'''
        rotated_piece = self._get_rotated_piece()
        min_x = min(cell[0] for cell in rotated_piece)
        max_x = max(cell[0] for cell in rotated_piece)
        return max_x - min_x + 1
    
    def _check_available_width(self, direction):
        rotated_piece = self._get_rotated_piece()

        for block in rotated_piece:
            x, y = block
            new_x = self.current_pos[0] + x + direction

            # Check if the new position is within the board boundaries
            if not (0 <= new_x < 10):
                return False

            # Check if the new position collides with existing blocks on the board
            if self.board[y + self.current_pos[1]][new_x] != 0:
                return False

        return True

    def _get_initial_board_props(self, initial_board):
        lines_left = self.lines_left
        moves_left = self.moves_left
        holes = self._number_of_holes(initial_board)
        total_bumpiness, max_bumpiness = self._bumpiness(initial_board)
        sum_height, max_height, min_height = self._height(initial_board)
        return [float(lines_left), float(moves_left), float(holes), float(total_bumpiness), float(max_height)]

    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        lines_left = self.lines_left - lines
        moves_left = self.moves_left - 1
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [float(lines_left), float(moves_left), float(holes), float(total_bumpiness), float(max_height)]

    def get_state_size(self):
        '''Size of the state'''
        return 5

    def get_next_states(self):
        '''Get all possible next states'''
        actions = []
        states = []
        boards = []
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
  
            piece = Tetris.TETROMINOS[piece_id][rotation]
     
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    boards.append(board)
                    states.append(self._get_board_props(board))
                    actions.append((x, rotation))

        return torch.tensor(states).type(torch.FloatTensor).cuda(), actions, boards
    
    def _draw_board(self, screen):
        surface = pygame.Surface(screen.get_size())
        surface.fill((255, 255, 255))

        # Clear the entire surface
        pygame.draw.rect(surface, (255, 255, 255), (0, 0, surface.get_width(), surface.get_height()))

        # Draw the Tetris board
        pygame.draw.rect(surface, (0, 0, 0, 128), (0, 0, Tetris.BOARD_WIDTH * self.block_size, Tetris.BOARD_HEIGHT * self.block_size), 3)
        for y, row in enumerate(self._get_complete_board()):
            for x, cell in enumerate(row):
                cell_rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(surface, Tetris.COLORS[cell], (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
                pygame.draw.rect(surface, (224, 224, 224), cell_rect, 1)

        # Draw the next piece
        pygame.draw.rect(surface, (224, 224, 224), ((Tetris.BOARD_WIDTH + 1) * self.block_size, self.block_size, 5 * self.block_size, 5 * self.block_size), 3)
        title_font = pygame.font.Font(None, 30)
        title_text = title_font.render("Next Piece", True, (0, 0, 0))
        title_pos = ((Tetris.BOARD_WIDTH * self.block_size + 50), self.block_size-20)
        surface.blit(title_text, title_pos)

        if self.next_piece in Tetris.TETROMINOS:
            next_piece_data = Tetris.TETROMINOS[self.next_piece][0]
            next_piece_color = Tetris.COLORS[Tetris.MAP_PLAYER]
            next_piece_width = len(next_piece_data[0]) if next_piece_data else 0
            next_piece_height = len(next_piece_data) if next_piece_data else 0
            next_piece_x = Tetris.BOARD_WIDTH + 1 + (6 - next_piece_width) // 4
            next_piece_y = 1 + (10 - next_piece_height) // 5

            for x, y in next_piece_data:
                pygame.draw.rect(surface, next_piece_color, ((x + next_piece_x) * self.block_size, (y + next_piece_y) * self.block_size, self.block_size, self.block_size))

        # Draw the level
        font = pygame.font.Font(None, 30)
        level_text = font.render(f"Level: {self.level + 1}", True, (0, 0, 0))
        level_pos = ((Tetris.BOARD_WIDTH * self.block_size + 20), (self.block_size * Tetris.BOARD_HEIGHT - level_text.get_height()) // 2 - 50)
        surface.blit(level_text, level_pos)

        # Draw additional information
        font = pygame.font.Font(None, 30)
        moves_left_text = font.render(f"Moves Left: {self.moves_left}", True, (0, 0, 0))
        lines_cleared_text = font.render(f"Lines Cleared: {self.lines_required - self.lines_left}/{self.lines_required}", True, (0, 0, 0))

        moves_left_pos = ((Tetris.BOARD_WIDTH * self.block_size + 20), (self.block_size * Tetris.BOARD_HEIGHT - moves_left_text.get_height()) // 2)
        lines_cleared_pos = ((Tetris.BOARD_WIDTH * self.block_size + 20), (self.block_size * Tetris.BOARD_HEIGHT - lines_cleared_text.get_height()) // 2 + 50)

        surface.blit(moves_left_text, moves_left_pos)
        surface.blit(lines_cleared_text, lines_cleared_pos)

        # Display the game over status
        if self.game_over:
            status_font = pygame.font.Font(None, 50)
            if self.moves_left == 0 and self.lines_left > 0:
                status_text = status_font.render("Mission Failed!", True, (0, 0, 0))
            elif self.moves_left >= 0 and self.lines_left <= 0:
                status_text = status_font.render("Mission Passed!", True, (0, 0, 0))
            elif self.moves_left > 0 and self.lines_left > 0:
                status_text = status_font.render("Wasted!", True, (0, 0, 0))

            status_pos = (Tetris.BOARD_WIDTH * self.block_size // 2 - status_text.get_width() // 2, 
                        Tetris.BOARD_HEIGHT * self.block_size // 2 + 50)

            surface.blit(status_text, status_pos)

        screen.blit(surface, (0, 0))
        pygame.display.flip()


    def play(self, x, rotation, board, episode, steps, max_render_steps, fps, last_frame_delay, log_dir, screen, render=False,
             render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation
        frame = 0

        if render:
            pygame.display.set_caption(f'Tetris - Episode {episode} - Step {steps}')

        # Drop piece
        if render and steps <= max_render_steps:

            while not self._check_collision(self._get_rotated_piece(), self.current_pos):
                self._draw_board(screen)
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            if self._check_available_width(-1):
                                self.current_pos[0] -= 1
                        elif event.key == pygame.K_RIGHT:
                                if self._check_available_width(1):
                                    self.current_pos[0] += 1 
                            
                        elif event.key == pygame.K_UP:
                            if self._check_available_width(1) and self._check_available_width(-1):
                                self.current_rotation = (self.current_rotation + 90) % 360
                        elif event.key == pygame.K_DOWN:
                            while not self._check_collision(self._get_rotated_piece(), (self.current_pos[0], self.current_pos[1] + 1)):
                                self.current_pos[1] += 1
                            self.current_pos[1] -= 1
                self.current_pos[1] += 1
                frame += 1
            self.current_pos[1] -= 1
            # Update board and calculate score
            self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)

        else:
            self.board = board

        lines_cleared, self.board = self._clear_lines(self.board)
        self.lines_left -= lines_cleared
        self.moves_left -= 1

        if render:
            self._draw_board(screen) 
            pygame.display.flip()
            time.sleep(1 / fps)

        if self.moves_left == 0 and self.lines_left > 0:
            self._new_round()
            score = (1 / self.lines_left ** 2) - 2
            self.game_over = True
            if render:
                self._draw_board(screen)  # Draw the Tetris board
                pygame.display.flip()
                time.sleep(1 / fps)
        elif self.moves_left >= 0 and self.lines_left <= 0:
            self._new_round()
            score = 4
            self.level_complete = True
            if render:
                self._draw_board(screen)  # Draw the Tetris board
                pygame.display.flip()
                time.sleep(1 / fps)
        elif self.moves_left > 0 and self.lines_left > 0:
            self._new_round()
            if self.game_over:
                score = -4
                if render:
                    self._draw_board(screen)  # Draw the Tetris board
                    pygame.display.flip()
                    time.sleep(1 / fps)
            else:
                score = 0

        # time.sleep(0.5)
        self.score += score
        return float(score), self.game_over, self.level_complete

    def render(self, episode, steps, max_render_steps, frame, fps, last_frame_delay, log_dir):
        '''Renders and saves the current board as a frame'''
        frames_save_dir = os.path.join(log_dir, 'gameplay.episode.{}'.format(episode))
        if not os.path.isdir(frames_save_dir):
            os.mkdir(frames_save_dir)

        rect_board = self.RECT_BOARD
        rect_piece = self.RECT_PIECE
        rect_text_1 = self.RECT_TEXT_1
        rect_text_2 = self.RECT_TEXT_2
        rect_text_3 = self.RECT_TEXT_3
        rect_text_4 = self.RECT_TEXT_4
        fig = plt.figure(figsize=(6, 6))

        img_board = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img_board = np.array(img_board).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img_board = img_board[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img_board = Image.fromarray(img_board, 'RGB')
        img_board = np.array(img_board)

        ax_board = fig.add_axes(rect_board)
        ax_board.imshow(img_board)
        ax_board.set_title('Game board')
        ax_board.set_xticklabels('')
        ax_board.set_yticklabels('')

        img_piece = [Tetris.COLORS[p]
                     for row in Tetris.PIECES_SHAPES[self.next_piece if self.next_piece >= 0 else -1]
                     for p in row]
        img_piece = np.array(img_piece).reshape(4, 6, 3).astype(np.uint8)
        img_piece = img_piece[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img_piece = Image.fromarray(img_piece, 'RGB')
        # img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img_piece = np.array(img_piece)

        # fig.canvas.draw()
        ax_piece = fig.add_axes(rect_piece)
        ax_piece.imshow(img_piece, interpolation='nearest')
        ax_piece.set_title('Next piece')
        ax_piece.set_xticklabels('')
        ax_piece.set_yticklabels('')

        ax_text_1 = fig.add_axes(rect_text_1)
        ax_text_1.axis('off')
        text_1 = 'Moves left: {}/{}'.format(self.moves_left, self.moves_required)
        ax_text_1.text(0, 0.5, text_1, ha='left', va='center', fontsize=12)

        ax_text_2 = fig.add_axes(rect_text_2)
        ax_text_2.axis('off')
        text_2 = 'Lines to clear: {}/{}'.format(self.lines_left, self.lines_required)
        ax_text_2.text(0, 0.5, text_2, ha='left', va='center', fontsize=12)

        image_dir = os.path.join(frames_save_dir, 'step.{}.frame.{}.png'.format(steps, frame))
        self.images_saved.append(image_dir)
        # frames.append(plt_to_numpy(fig))

        plt.savefig(image_dir)
        self.frames.append(imageio.imread(image_dir))

        if not self.game_over:
            plt.close()
        else:
            if self.moves_left >= 0 and self.lines_left <= 0:
                status = 0
            elif self.moves_left == 0 and self.lines_left > 0:
                status = 1
            elif self.moves_left > 0 and self.lines_left > 0:
                status = 2

            ax_text_3 = fig.add_axes(rect_text_3)
            ax_text_3.axis('off')
            text_3 = self.GAMEOVER_TAG_1[status]
            ax_text_3.text(0.5, 0.5, text_3, ha='center', va='center', fontsize=16)

            ax_text_4 = fig.add_axes(rect_text_4)
            ax_text_4.axis('off')
            text_4 = self.GAMEOVER_TAG_2[status]
            ax_text_4.text(0.5, 0.5, text_4, ha='center', va='center', fontsize=16)

            image_dir = os.path.join(frames_save_dir, 'step.{}.frame.gameover.png'.format(steps, frame))
            self.images_saved.append(image_dir)
            # frames.append(plt_to_numpy(fig))

            plt.savefig(image_dir)
            plt.close()

            for _ in range(fps * last_frame_delay):
                self.frames.append(imageio.imread(image_dir))

            imageio.mimsave(os.path.join(log_dir, 'episode.{}.gameplay.animation.gif'.format(episode)), self.frames, fps=fps)

            try:
                for image in self.images_saved:
                    if os.path.exists(image):
                        os.remove(image)
            except OSError as err:
                print(err)
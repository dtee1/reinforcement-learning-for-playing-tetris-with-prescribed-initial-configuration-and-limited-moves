import random

import numpy as np
class TetrisGameGenerator:
    tetromino_shapes = {
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'O': [[[1, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }
    def __init__(self, height=20, width=10, seed=None, goal=15, tetrominoes=40, initial_height_max=7):
        self.height = height
        self.width = width
        self.seed = seed
        self.goal = goal
        self.tetrominoes = tetrominoes
        self.initial_height_max = initial_height_max
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.winnable = False
        self.tetrominoes_names = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

        # the higher the density the more compact the tetrominoes will be placed 0-9
        self.density = 9

        random.seed(self.seed)
        self.fill_grid()
        self.sequence = self.generate_tetromino_sequence(self.tetrominoes)

    def rotate_tetromino(self, tetromino, rotation):
        return tetromino[rotation % len(tetromino)]

    def is_valid_move(self, tetromino, row, col):
        shape = np.array(tetromino)
        rows, cols = shape.shape

        if (
            row + rows > self.height or
            col < 0 or col + cols > self.width
        ):
            return False

        overlap = self.board[row:row+rows, col:col+cols] + shape
        return not np.any(overlap > 1)

    def place_tetromino(self, tetromino, row, col):
        shape = np.array(tetromino)
        rows, cols = shape.shape

        while row + rows <= self.height and not np.any(self.board[row:row+rows, col:col+cols] + shape > 1):
            row += 1

        self.board[row-1:row-1+rows, col:col+cols] += shape

        self.clear_lines()

    def clear_lines(self):
        full_rows = np.all(self.board, axis=1)

        self.board = np.vstack([np.zeros((np.sum(full_rows), self.width), dtype=int), self.board[~full_rows]])

    def evaluate_columns(self, tetromino):
        columns_to_try = list(range(self.width - len(tetromino[0]) + 1))
        columns_to_try.sort(key=lambda col: -self.calculate_placement_height(tetromino, col))

        return columns_to_try

    def calculate_placement_height(self, tetromino, col):
        shape = np.array(tetromino)
        rows, cols = shape.shape

        height = 0
        while height + rows <= self.height and not np.any(self.board[height:height+rows, col:col+cols] + shape > 1):
            height += 1

        return height


    def fill_grid(self):
        for _ in range(40):
            tetromino = random.choice(self.tetrominoes_names)
            rotation = random.randint(0, len(self.tetromino_shapes[tetromino]) - 1)
            tetromino = self.rotate_tetromino(self.tetromino_shapes[tetromino], rotation)
            col_to_try = self.evaluate_columns(tetromino)[9 - self.density]
            if(self.is_valid_move(tetromino, 0, col_to_try)):
                if(self.height + 1 - self.calculate_placement_height(tetromino, col_to_try) <= self.initial_height_max):
                    self.place_tetromino(tetromino, 0, col_to_try)
                else:
                    break



    def generate_tetromino_sequence(self, max_moves=None):
        tetromino_names = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        bag_size = 7
        bags = []

        while True:
            bag = tetromino_names.copy()
            random.shuffle(bag)

            while any(bag[i] == bag[i + 1] in ['S', 'Z'] for i in range(bag_size - 1)):
                random.shuffle(bag)

            bags.append(bag)

            if len(bags) * bag_size >= max_moves:
                sequence = [tetromino for bag in bags for tetromino in bag]
                return sequence[:max_moves] if max_moves else sequence

    def print_grid(self):
        for row in range(self.height):
            for col in range(self.width):
                print(self.board[row][col], end=' ')
            print()


if __name__ == '__main__':
    game = TetrisGameGenerator(seed=1, goal=10, tetrominoes=40)
    game.print_grid()
    print(game.sequence)

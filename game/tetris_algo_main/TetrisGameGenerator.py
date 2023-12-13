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
        self.tetrominoes_names = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

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


    def calculate_placement_height(self, tetromino, col):
        shape = np.array(tetromino)
        rows, cols = shape.shape

        height = 0
        while height + rows <= self.height and not np.any(self.board[height:height+rows, col:col+cols] + shape > 1):
            height += 1

        return height


    def fill_grid(self):
        while True:
            # Randomly select a tetromino
            tetromino = random.choice(self.tetrominoes_names)
            # Randomly select a rotation
            rotation = random.randint(0, len(self.tetromino_shapes[tetromino]) - 1)
            shape = self.rotate_tetromino(self.tetromino_shapes[tetromino], rotation)
            col_to_try = random.randint(0, self.width - len(shape[0]))
            if self.is_valid_move(shape, 0, col_to_try):
                placement_height = self.calculate_placement_height(shape, col_to_try)
                if self.height + 1 - placement_height <= self.initial_height_max:
                    self.place_tetromino(shape, 0, col_to_try)
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


def generate_board_and_sequence(seed, tetrominoes, initial_height_max, goal=0):
    game = TetrisGameGenerator(seed=seed, goal=goal, tetrominoes=tetrominoes,initial_height_max= initial_height_max)
    return game.board, game.sequence




def main():
    # for i in range(1000):
    #     board, sequence = generate_board_and_sequence(i, 40, 7)

    game = TetrisGameGenerator(seed=15, goal=15, tetrominoes=40, initial_height_max=7)
    game.print_grid()
    print(game.sequence)

if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', sort='cumulative')

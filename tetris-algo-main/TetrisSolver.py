import numpy as np
from collections import deque

class TetrisSolver:
    tetromino_shapes = {
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'O': [[[1, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }



    def __init__(self, board, sequence, goal, max_attempts=100000):
        self.board = np.array(board)
        self.initial_board = np.array(board)
        self.height = len(board)
        self.width = len(board[0])
        self.sequence = deque(sequence)
        self.lines_cleared = 0
        self.stack = []
        self.failed_attempts = 0
        self.goal = goal
        self.max_attempts = max_attempts

        self.additional_max_attempts = True

    def reset(self):
        self.board = np.copy(self.initial_board)
        self.lines_cleared = 0
        self.stack = []
        self.failed_attempts = 0

    def rotate_tetromino(self, tetromino, rotation):
        return tetromino[rotation % len(tetromino)]

    def is_valid_move(self, tetromino, row, col):
        shape = np.asarray(tetromino)
        rows, cols = self.get_tetromino_dimensions(shape)

        if (
            row + rows > self.height or
            col < 0 or col + cols > self.width
        ):
            return False

        for r in range(rows):
            for c in range(cols):
                if shape[r][c] == 1 and self.board[row+r][col+c] == 1:
                    return False

        return True

    def get_tetromino_dimensions(self, tetromino):
        if hasattr(tetromino, 'rows') and hasattr(tetromino, 'cols'):
            return tetromino.rows, tetromino.cols

        rows, cols = tetromino.shape
        return rows, cols


    def place_tetromino(self, tetromino, row, col):
        shape = np.asarray(tetromino)
        rows, cols = shape.shape

        while row + rows <= self.height and not np.any(np.add(self.board[row:row+rows, col:col+cols], shape) > 1):
            row += 1

        np.add(self.board[row-1:row-1+rows, col:col+cols], shape, out=self.board[row-1:row-1+rows, col:col+cols])

        self.clear_lines()

    def clear_lines(self):
        full_rows = np.all(self.board, axis=1)
        self.lines_cleared += np.sum(full_rows)

        self.board = np.vstack([np.zeros((np.sum(full_rows), self.width), dtype=int), self.board[~full_rows]])

    def visualize(self, board=None):
        if board is None:
            board = self.board

        return '\n'.join([' '.join(map(str, row)) for row in board])

    def is_game_over(self):
        return any(self.board[0][col] == 1 for col in range(self.width))

    def evaluate_columns(self, tetromino):
        columns_to_try = list(range(self.width - len(tetromino[0]) + 1))
        columns_to_try.sort(key=lambda col: -self.calculate_placement_height(tetromino, col))
        return columns_to_try[0]

    def calculate_placement_height(self, tetromino, col):
        shape = np.array(tetromino)
        rows, cols = shape.shape

        height = 0
        while height + rows <= self.height and not np.any(self.board[height:height+rows, col:col+cols] + shape > 1):
            height += 1

        return height


    def solve(self, current=None):
        current = current if current else self.sequence.popleft()
        shape = self.tetromino_shapes[current]

        for rotation in range(len(shape)):
            columns_to_try = list(range(self.width - len(shape[0]) + 1))

            for _ in columns_to_try:
                if self.failed_attempts >= self.max_attempts:
                    if self.additional_max_attempts and self.goal - self.lines_cleared <= 2:
                        self.max_attempts += self.max_attempts
                        self.additional_max_attempts = False
                        continue

                    return False, self.stack, self.failed_attempts
                boardcopy = np.copy(self.board)
                col = self.evaluate_columns(shape[rotation])

                current_iteration_lines_cleared = self.lines_cleared
                if self.is_valid_move(shape[rotation], 0, col):
                    self.place_tetromino(shape[rotation], 0, col)
                else:
                    self.failed_attempts += 1
                    continue

                if self.is_game_over():
                    self.board = np.copy(boardcopy)
                    self.lines_cleared = current_iteration_lines_cleared
                    self.failed_attempts += 1
                    continue

                elif self.lines_cleared >= self.goal:
                    self.stack.append((current, rotation, col))
                    return True, self.stack, self.failed_attempts

                elif self.sequence:
                    self.stack.append((current, rotation, col))
                    next_tetromino = self.sequence.popleft()
                    result, stack, attempts = self.solve(next_tetromino)
                    if result:
                        return True, stack, attempts
                    self.sequence.appendleft(next_tetromino)
                    self.stack.pop()
                    self.lines_cleared = current_iteration_lines_cleared
                    self.board = np.copy(boardcopy)

                else:
                    self.board = np.copy(boardcopy)
                    self.lines_cleared = current_iteration_lines_cleared
                    self.failed_attempts += 1

                if(rotation == len(current) - 1 and col == self.width - len(shape[rotation][0])):
                    self.failed_attempts += 1
                    self.board = np.copy(boardcopy)
                    self.lines_cleared = current_iteration_lines_cleared

        return False, self.stack, self.failed_attempts

    def visualize_moves(self, stack):
        self.reset()
        for tetromino, rotation, col in stack:
            initial_lines_cleared = self.lines_cleared
            self.place_tetromino(self.tetromino_shapes[tetromino][rotation], 0, col)
            print("Tetromino: ", tetromino, " Rotation: ", rotation, " Column: ", col)
            print("Lines cleared: ", self.lines_cleared - initial_lines_cleared)
            print(self.visualize())
            print()



    def evaluate_board(self, board):
        return np.sum(board)

    def test(self, tetrominoes, positions):
        positions = deque(positions)
        for tetromino in tetrominoes:
            col = positions.popleft()

            self.place_tetromino(tetromino, 0, col)
            if(self.is_game_over()):
                return False
            board = np.copy(self.board)
            self.visualize(board)

        lines_cleared = self.lines_cleared
        self.lines_cleared = 0
        self.board = np.zeros((self.height, self.width), dtype=int)
        return lines_cleared

if __name__ == '__main__' :
    from time import time
    from TetrisGameGenerator import TetrisGameGenerator
    import cProfile

    height = 20
    width = 10

    seed = 1
    goal = 5
    tetrominoes = 30

    game = TetrisGameGenerator(height=height, width=width, seed=seed, goal=goal, tetrominoes=tetrominoes)
    board = game.board
    sequence = game.sequence



    solver = TetrisSolver(board, sequence, goal)

    print(solver.visualize())

    profiler = cProfile.Profile()
    profiler.enable()

    start = time()
    result, stack, failed_attempts = solver.solve()
    end = time()

    profiler.disable()
    profiler.print_stats(sort='cumulative')

    print(game.sequence)
    print('Time taken: ', end - start)
    print('Result: ', result)
    print('Stack: ', stack)
    print('Failed attempts: ', failed_attempts)

    solver.visualize_moves(stack)



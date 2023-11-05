import numpy as np
from random import randint, getrandbits

tetrominos = (
    np.array(((False,False,False,False),
    (True,True,True,True),
    (False,False,False,False),
    (False,False,False,False)), dtype=bool),

    np.array(((True,True),
    (True,True)), dtype=bool),

    np.array(((False,True,False),
    (True,True,True),
    (False,False,False)), dtype=bool),

    np.array(((False,True,True),
    (True,True,False),
    (False,False,False)), dtype=bool),

    np.array(((True,True,False),
    (False,True,True),
    (False,False,False)), dtype=bool),

    np.array(((True,False,False),
    (True,True,True),
    (False,False,False)), dtype=bool),
    
    np.array(((False,False,True),
    (True,True,True),
    (False,False,False)), dtype=bool)
)

class Tetris:
    def __init__(self, L: int, M: int, random_pieces=False):
        self.board, self.pieces = self.__get_initial_config()
        self.L = L
        self.M = M
        self.lines_cleared = 0
        self.moves_used = 0
        self.state = None
        self.random_pieces = random_pieces

    def __get_initial_config(self):
        return np.full((20, 10), False, dtype=bool), [randint(0, 6) for _ in range(2)]
    
    def get_info(self):
        return self.board, self.pieces[0], self.pieces[1], self.L - self.lines_cleared, self.M - self.moves_used, self.state

    def move(self, rotations: int, location: int):
        piece = self.pieces.pop(0)
        if self.random_pieces:
            self.pieces.append(bool(getrandbits(1)))

        tetromino = tetrominos[piece]

        # Rotate the piece counterclockwise
        tetromino = np.rot90(tetromino, rotations)

        # Truncate the tetromino
        rows_with_true = np.any(tetromino, axis=1)
        cols_with_true = np.any(tetromino, axis=0)
        tetromino = tetromino[rows_with_true][:, cols_with_true]

        # Ensure the location is not out of bounds horizontally
        tetromino_width = tetromino.shape[1]
        location = min(location, 10-tetromino_width)

        # Calculate drop
        reverse_tetromino_topography = []
        for col in tetromino.T:
            reverse_tetromino_topography.append(np.where(col)[0][-1])

        board_topography = []
        for col in self.board.T[location:location+tetromino_width, :]:
            result = np.where(col)[0]
            board_topography.append(result[0] if len(result) != 0 else 20)

        drop_deltas = np.array(board_topography) - np.array(reverse_tetromino_topography)
        drop = min(drop_deltas) - 1

        # Check if out of bounds
        if drop < 0:
            self.state = False
            return False

        # Insert Block
        self.board[drop:drop + tetromino.shape[0], location:location + tetromino_width] = tetromino

        self.moves_used += 1

        # Clear lines
        rows_all_trues = np.all(self.board[drop:drop + tetromino.shape[0], :], axis=1)

        rows_cleared = np.count_nonzero(rows_all_trues)
        
        # If no lines cleared
        if rows_cleared == 0:
            if self.moves_used >= self.M:
                self.state = False
            return False
        
        indices_to_clear = []

        for row_index, row in enumerate(rows_all_trues):
            if row:
                indices_to_clear.append(drop + row_index)

        indices_left = [x for x in range(0, 20) if x not in indices_to_clear]

        self.board = self.board[indices_left]
        rows_to_add = np.full((rows_cleared, 10), False)
        self.board = np.vstack((rows_to_add, self.board))

        self.lines_cleared -= rows_cleared

        if self.lines_cleared >= self.L:
            self.state = True

        if self.moves_used >= self.M:
                self.state = False
    def reset(self):
        self.__init__(self.L, self.M, random_pieces=self.random_pieces)

import numpy as np

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
    def __init__(self, L: int, M: int):
        self.board, self.pieces = self.__get_initial_config()
        self.L = L
        self.M = M

    def __get_initial_config(self):
        return np.full((20, 10), False, dtype=bool), [6,0,0,0]
    
    def get_info(self):
        return self.board, self.pieces[0], self.pieces[1]

    def move(self, rotations: int, location: int):
        piece = self.pieces.pop(0)
        tetromino = tetrominos[piece]

        # Rotate the piece counterclockwise
        tetromino = np.rot90(tetromino, rotations)

        # Truncate the tetromino
        rows_with_true = np.any(tetromino, axis=1)
        cols_with_true = np.any(tetromino, axis=0)
        tetromino = tetromino[rows_with_true][:, cols_with_true]

        # Ensures the location is not out of bounds horizontally
        tetromino_width = tetromino.shape[1]
        location = min(location, 10-tetromino_width)

        reverse_tetromino_topography = []
        for col in tetromino.T:
            reverse_tetromino_topography.append(np.where(col)[0][-1])

        board_topography = []
        for col in self.board.T[location:location+tetromino_width, :]:
            result = np.where(col)[0]
            board_topography.append(result[0] if result else 20)

        drop_topography = np.array(board_topography) - np.array(reverse_tetromino_topography)

import numpy as np
from random import randint, getrandbits
import time

tetrominos = (
    (
        (np.array(((True,True,True,True),), dtype=bool), (0,0,0,0)),
        (np.array(((True,),(True,),(True,),(True,)), dtype=bool), (3,))
    ),
    (
        (np.array(((False,False,True),(True,True,True)), dtype=bool), (1,1,1)),
        (np.array(((True,True),(False,True),(False,True)), dtype=bool), (0,2)),
        (np.array(((True,True,True),(True,False,False)), dtype=bool), (1,0,0)),
        (np.array(((True,False),(True,False),(True,True)), dtype=bool), (2,2))
    ),
    (
        (np.array(((True,False,False),(True,True,True)), dtype=bool), (1,1,1)),
        (np.array(((False,True),(False,True),(True,True)), dtype=bool), (2,2)),
        (np.array(((True,True,True),(False,False,True)), dtype=bool), (0,0,1)),
        (np.array(((True,True),(True,False),(True,False)), dtype=bool), (2,0))
    ),
    (
        (np.array(((False,True,False),(True,True,True)), dtype=bool), (1,1,1)),
        (np.array(((False,True),(True,True),(False,True)), dtype=bool), (1,2)),
        (np.array(((True,True,True),(False,True,False)), dtype=bool), (0,1,0)),
        (np.array(((True,False),(True,True),(True,False)), dtype=bool), (2,1))
    ),
    (
        (np.array(((False,True,True),(True,True,False)), dtype=bool), (1,1,0)),
        (np.array(((True,False),(True,True),(False,True)), dtype=bool), (1,2))
    ),
    (
        (np.array(((True,True,False),(False,True,True)), dtype=bool), (0,1,1)),
        (np.array(((False,True),(True,True),(True,False)), dtype=bool), (2,1))
    ),
    (
        (np.array(((True,True),(True,True)), dtype=bool), (1,1)),
    )
)

def get_tetromino(piece: int, rotations: int) -> tuple[np.array, tuple[int, ...]]:
    return tetrominos[piece][rotations % len(tetrominos[piece])]

class Tetris:
    def __init__(self, L: int, M: int, random_pieces=False):
        self.L = L
        self.M = M
        self.random_pieces = random_pieces

        while True:
            self.board = None
            self.pieces = []

            if self.__get_initial_config():
                break
        self.lines_cleared = 0
        self.moves_used = 0
        self.state = None
        
    # Carving Approach
    def __get_initial_config(self) -> None:
        # Initial full board generation
        initial_empty = 20 - self.L # randint(0, 20 - self.L)

        self.board = np.full((20, 10), True, dtype=bool)
        self.board[:initial_empty, :] = False

        pieces = list(range(7))
        checkpoints = [np.copy(self.board)]

        checkpoint_loop = 0
        revert_loop = 0

        while (len(self.pieces) < self.M) and np.all(self.board[-1]):
            # Choose a random piece from the pieces left
            piece_index = randint(0, len(pieces) - 1)
            piece = pieces[piece_index]
            rotations = randint(0, 3)
            tetromino_width = get_tetromino(piece, rotations)[0].shape[1]
            location = randint(0, 10-tetromino_width)

            if self.carve(piece, rotations, location):
                checkpoint_loop = 0
                self.pieces.insert(0, piece)
                del pieces[piece_index]
                if not pieces:
                    revert_loop = 0
                    checkpoints.append(np.copy(self.board))
                    pieces = list(range(7))
            else:
                if revert_loop > 100:
                    if len(checkpoints) > 1:
                        # Delete the last checkpoint
                        checkpoints.pop()
                        # Load from the new last checkpoint
                        self.board = np.copy(checkpoints[-1])
                        self.pieces = self.pieces[-((len(checkpoints) - 1) * 7):]
                        pieces = list(range(7))
                    
                    checkpoint_loop = 0
                    revert_loop = 0
                elif checkpoint_loop > 100:
                    # Load from the last checkpoint
                    self.board = np.copy(checkpoints[-1])
                    self.pieces = self.pieces[-((len(checkpoints) - 1) * 7):]
                    pieces = list(range(7))

                    checkpoint_loop = 0
                    revert_loop += 1
                else:
                    checkpoint_loop += 1

        return True

    def carve(self, piece: int, rotations: int, location: int) -> bool:
        tetromino, reverse_tetromino_topography = get_tetromino(piece, rotations)
        tetromino_height, tetromino_width = tetromino.shape

        # Calculate drop location
        drop_deltas = self.calculate_drop_deltas(location, reverse_tetromino_topography, tetromino_width)
        drop = self.calculate_drop(drop_deltas)
        push = reverse_tetromino_topography[np.argmin(drop_deltas)] + 1

        drop += push

        # If the drop goes out of bounds then the carve failed
        if drop + tetromino_height > 20:
            return False

        # Calculate overlap
        board_slice = self.board[drop:drop + tetromino_height, location:location + tetromino_width]
        overlap = np.all(np.logical_not(tetromino) | board_slice)

        # If no overlap then carving failed
        if not overlap:
            return False
        
        # Apply the carve
        self.board[drop:drop + tetromino_height, location:location + tetromino_width] ^= tetromino
        
        # Make a new drop to ensure it falls where the carve was
        # Ensures that there were no blocking blocks above and that there were supporting blocks below
        new_drop_deltas = self.calculate_drop_deltas(location, reverse_tetromino_topography, tetromino_width)
        new_drop = self.calculate_drop(new_drop_deltas)
        
        # If the new drop does not land where the carve was done then revert the carve and carving failed
        if new_drop != drop:
            self.board[drop:drop + tetromino_height, location:location + tetromino_width] |= tetromino
            return False

        # Carving succeeded
        return True

    def move(self, rotations: int, location: int) -> None:
        piece = self.pieces.pop(0)
        if self.random_pieces:
            self.pieces.append(bool(getrandbits(1)))

        tetromino, reverse_tetromino_topography = get_tetromino(piece, rotations)

        # Ensure the location is not out of bounds horizontally
        tetromino_width = tetromino.shape[1]
        location = min(location, 10-tetromino_width)
        
        drop_deltas = self.calculate_drop_deltas(location, reverse_tetromino_topography, tetromino_width)
        drop = self.calculate_drop(drop_deltas)

        # Check if out of bounds
        if drop < 0:
            self.state = False
            return

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
            return
        
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
    
    def calculate_drop(self, drop_deltas) -> int:
        return min(drop_deltas) - 1
    
    def calculate_drop_deltas(self, location, reverse_tetromino_topography, tetromino_width):
        board_topography = []
        for col in self.board.T[location:location+tetromino_width, :]:
            result = np.where(col)[0]
            board_topography.append(result[0] if len(result) != 0 else 20)

        return np.array(board_topography) - np.array(reverse_tetromino_topography)

    def get_state(self) -> tuple[np.array, int, int, int, int, bool]:
        return self.board, self.pieces[0], self.pieces[1], self.L - self.lines_cleared, self.M - self.moves_used, self.state
            
    def reset(self) -> None:
        self.__init__(self.L, self.M, random_pieces=self.random_pieces)

start_time = time.time()
iterations = 100
for i in range(iterations):
    game = Tetris(15, 40)
print(iterations / (time.time() - start_time))
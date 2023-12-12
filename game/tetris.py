import numpy as np
import random
import time
from tqdm import tqdm

from typing import Callable

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

class RandomPieceGenerator:
    def __init__(self) -> None:
        self.pieces = []
        self.last_generated_random_piece_index = None

    def generate_pieces(self) -> None:
        self.pieces = list(range(7))

    def _regenerate(method: Callable) -> Callable[..., tuple[any, bool]]:
        def wrapper(self, *args, **kwargs) -> tuple[any, bool]:
            regenerated = False

            if not self.pieces:
                self.generate_pieces()
                regenerated = True
            
            method_result = method(self, *args, **kwargs)

            return method_result, regenerated
        return wrapper

    @_regenerate
    def get_random_piece(self):
        self.last_generated_random_piece_index = random.randint(0, len(self.pieces) - 1)
        return self.pieces[self.last_generated_random_piece_index]

    def delete_last_generated_random_piece(self):
        del self.pieces[self.last_generated_random_piece_index]
        
    @_regenerate
    def _generate_sequence(self) -> None:
        random.shuffle(self.pieces)

    def get_random_sequence(self, length: int) -> list[int]:
        sequence = []
        while len(sequence) < length:
            self._generate_sequence()
            sequence.extend(self.pieces[:min(length - len(sequence), 7)])
            self.pieces = []

        return sequence
    
    def __len__(self) -> int:
        return len(self.pieces)
    
class CheckpointManager:
    def __init__(self, initial_checkpoint: any) -> None:
        self.checkpoints = [initial_checkpoint]

        self.attempts = 0
        self.max_attempts = 40

        self.checkpoint_uses = 0
        self.max_checkpoint_uses = 10

    def add_attempt(self) -> bool:
        self.attempts += 1
        return self.attempts > self.max_attempts
    
    def add_checkpoint(self, checkpoint: any) -> None:
        self.checkpoints.append(checkpoint)
    
    def load_checkpoint(self) -> any:
        self.attempts = 0

        if len(self.checkpoints) > 1 and self.checkpoint_uses > self.max_checkpoint_uses:
            del self.checkpoints[-1]
            reverted = True
            self.checkpoint_uses = 0
        else:
            reverted = False
            self.checkpoint_uses += 1

        return self.checkpoints[-1], reverted

class Tetris:
    def __init__(self, L: int, M: int, random_pieces=False, max_revert=10, max_checkpoint=40):
        # Create the random piece generator instance
        self.random_piece_generator = RandomPieceGenerator()

        self.L = L
        self.M = M

        self.random_pieces = random_pieces
        self.max_revert = max_revert
        self.max_checkpoint = max_checkpoint

        self.warm_reset = True

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
        # Calculate the number of lines that need to be empty at the top
        initial_empty = 20 - self.L # random.randint(0, 20 - self.L)

        # Generate a completely full board
        self.board = np.full((20, 10), True, dtype=bool)

        # Empty out the needed number of lines at the top
        self.board[:initial_empty, :] = False

        # Create the checkpoint manager instance
        checkpoint_manager = CheckpointManager(np.copy(self.board))

        while (len(self.pieces) < self.M) and np.count_nonzero(self.board[-1]) >= len(self.board[-1]) / 2:
            # Get a random piece from the generator
            random_piece, regenerated = self.random_piece_generator.get_random_piece()

            if regenerated:
                checkpoint_manager.add_checkpoint(np.copy(self.board))

            rotations = random.randint(0, 3)
            tetromino_width = get_tetromino(random_piece, rotations)[0].shape[1]
            location = random.randint(0, 10-tetromino_width)
                        
            if self.carve(random_piece, rotations, location, len(self.random_piece_generator) == 7):
                self.pieces.insert(0, random_piece)
                self.random_piece_generator.delete_last_generated_random_piece()
            else:
                if checkpoint_manager.add_attempt():
                    checkpoint, reverted = checkpoint_manager.load_checkpoint()
                    self.board = np.copy(checkpoint)
                    self.pieces = self.pieces[(14 if reverted else 7) - len(self.random_piece_generator):]
                    
                    
                    self.random_piece_generator.generate_pieces()
        
        # If less pieces were used than the allows maximum then pad out the pieces randomly
        if (len(self.pieces) < self.M):
            self.pieces.extend(self.random_piece_generator.get_random_sequence(self.M - len(self.pieces)))

        return True

    def carve(self, piece: int, rotations: int, location: int, allow_partial: bool) -> bool:
        tetromino, reverse_tetromino_topography = get_tetromino(piece, rotations)
        tetromino_height, tetromino_width = tetromino.shape

        # Calculate drop location
        drop_deltas = self.calculate_drop_deltas(location, reverse_tetromino_topography, tetromino_width)
        drop = self.calculate_drop(drop_deltas)
        push = reverse_tetromino_topography[np.argmin(drop_deltas)] + 1

        drop += push

        increments = tetromino_height if allow_partial else 1
        for increment in range(increments):
            print(drop)
            if self.calculate_carve(drop, location, tetromino, reverse_tetromino_topography, allow_partial):
                return True
            drop -= 1
        return False

    def calculate_carve(self, drop: int, location: int, tetromino: np.array, reverse_tetromino_topography: tuple[int, ...], allow_partial: bool):
        tetromino_height, tetromino_width = tetromino.shape

        # If the drop goes out of bounds then the carve failed
        if drop + tetromino_height > 20:
            return False

        # Calculate overlap
        if not allow_partial:            
            board_slice = self.board[drop:drop + tetromino_height, location:location + tetromino_width]
            overlap = np.all(np.logical_not(tetromino) | board_slice)

            # If no overlap then carving failed
            if not overlap:
                return False
        
        # Apply the carve
        self.board[drop:drop + tetromino_height, location:location + tetromino_width] &= np.logical_not(tetromino)
        
        if not allow_partial:
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
            self.pieces.append(bool(random.getrandbits(1)))

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
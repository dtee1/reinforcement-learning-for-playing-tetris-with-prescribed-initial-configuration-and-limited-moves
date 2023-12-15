import numpy as np
import random
import multiprocessing
from typing import Callable

from tetris_algo_main import main

piece_translations = {
    'I': 0,
    'L': 1,
    'J': 2,
    'T': 3,
    'S': 4,
    'Z': 5,
    'O': 6
}


def translate(batch: list):
    return [(game.board.astype(bool), [random.randint(0, 6)] + [piece_translations[letter] for letter in game.sequence][::-1]) for game in batch]


tetrominos = (
    (
        (np.array(((True, True, True, True),), dtype=bool), (0, 0, 0, 0)),
        (np.array(((True,), (True,), (True,), (True,)), dtype=bool), (3,))
    ),
    (
        (np.array(((False, False, True), (True, True, True)), dtype=bool), (1, 1, 1)),
        (np.array(((True, True), (False, True), (False, True)), dtype=bool), (0, 2)),
        (np.array(((True, True, True), (True, False, False)), dtype=bool), (1, 0, 0)),
        (np.array(((True, False), (True, False), (True, True)), dtype=bool), (2, 2))
    ),
    (
        (np.array(((True, False, False), (True, True, True)), dtype=bool), (1, 1, 1)),
        (np.array(((False, True), (False, True), (True, True)), dtype=bool), (2, 2)),
        (np.array(((True, True, True), (False, False, True)), dtype=bool), (0, 0, 1)),
        (np.array(((True, True), (True, False), (True, False)), dtype=bool), (2, 0))
    ),
    (
        (np.array(((False, True, False), (True, True, True)), dtype=bool), (1, 1, 1)),
        (np.array(((False, True), (True, True), (False, True)), dtype=bool), (1, 2)),
        (np.array(((True, True, True), (False, True, False)), dtype=bool), (0, 1, 0)),
        (np.array(((True, False), (True, True), (True, False)), dtype=bool), (2, 1))
    ),
    (
        (np.array(((False, True, True), (True, True, False)), dtype=bool), (1, 1, 0)),
        (np.array(((True, False), (True, True), (False, True)), dtype=bool), (1, 2))
    ),
    (
        (np.array(((True, True, False), (False, True, True)), dtype=bool), (0, 1, 1)),
        (np.array(((False, True), (True, True), (True, False)), dtype=bool), (2, 1))
    ),
    (
        (np.array(((True, True), (True, True)), dtype=bool), (1, 1)),
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
        self.last_generated_random_piece_index = random.randint(
            0, len(self.pieces) - 1)
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

    def reset(self) -> None:
        self.pieces.clear()
        self.last_generated_random_piece_index = None

    def __len__(self) -> int:
        return len(self.pieces)


class CheckpointManager:
    def __init__(self) -> None:
        self.checkpoints = []

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
    def __init__(self, L: int, M: int, warm_reset: bool = True, render: bool = False, framerate: int = 30, debug: bool = False):
        # Hyperparameters
        self.L: int = L
        self.M: int = M
        self.warm_reset = warm_reset
        self.render = render

        # State variables
        self.lines_cleared = 0
        self.moves_used = 0
        self.state = None

        # Debug variables
        self.debug = debug
        if self.debug:
            self.solution: list[tuple[int, int]] = []

        if render:
            import pygame
            self.pygame = pygame

            # Constants
            self.WIDTH, self.HEIGHT = 400, 800
            self.GRID_SIZE = 40

            self.PIECE_COLOR = (99, 64, 247)
            self.BOARD_COLOR = (255, 255, 255)

            # Initialize Pygame
            self.pygame.init()

            # Set up the game window
            self.screen = self.pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT))
            self.pygame.display.set_caption("Tetris")

            # Clock to control the frame rate
            self.clock = self.pygame.time.Clock()
            self.framerate = 30

            # Disable warm reset
            self.warm_reset = False

        # Components
        self.random_piece_generator: RandomPieceGenerator = RandomPieceGenerator()
        self.board: np.array = np.full((20, 10), False, dtype=bool)
        self.pieces: list[int] = []

        # Warm reset components
        if self.warm_reset:
            # Create the terminate event
            self.terminate_event = multiprocessing.Event()

            # Create the queue that will store future initial configurations
            self.queue = multiprocessing.Queue(maxsize=20)

            # Create the game instance that will be used by the warm reset worker
            warm_reset_tetris = Tetris(
                self.L, self.M, warm_reset=False, render=False)

            # Create the warm reset worker
            self.warm_reset_process = multiprocessing.Process(
                target=warm_reset_worker, args=(self.queue, warm_reset_tetris, self.terminate_event))

            # Create the forward-mode warm reset worker
            self.forward_warm_reset_process = multiprocessing.Process(
                target=forward_warm_reset_worker, args=(self.queue, self.terminate_event, self.L, self.M))

            # Start both workers
            self.warm_reset_process.start()
            self.forward_warm_reset_process.start()

        # Initiate a reset
        self.load_warm_reset()

    def render_frame(self, board: np.array):
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                color = self.PIECE_COLOR if board[y, x] else self.BOARD_COLOR
                self.pygame.draw.rect(self.screen, color, (x * self.GRID_SIZE,
                                                           y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        self.pygame.display.flip()
        self.clock.tick(self.framerate)

    # Carving Approach
    def _generate_initial_config(self) -> None:
        # Add the number of lines to clear as full lines onto the board
        self.board[-self.L:, :] = True

        # Create the checkpoint manager instance
        checkpoint_manager = CheckpointManager()

        # While a piece sequence hasn't been found and the last line hasn't been broken
        while (len(self.pieces) < self.M) and np.count_nonzero(self.board[-1]) > 8:
            # Get a random piece from the generator
            random_piece, regenerated = self.random_piece_generator.get_random_piece()

            # If the generator regenerated then add a checkpoint
            if regenerated:
                checkpoint_manager.add_checkpoint(np.copy(self.board))

            # Get some extra information
            rotations = random.randint(0, 3)
            tetromino_width = get_tetromino(
                random_piece, rotations)[0].shape[1]
            location = random.randint(0, 10-tetromino_width)

            # Attempt a carve
            # If it is successful then add the piece to the sequence
            if self.carve(random_piece, rotations, location, len(self.random_piece_generator) == 7):
                self.pieces.append(random_piece)
                if self.debug:
                    self.solution.append((rotations, location))
                self.random_piece_generator.delete_last_generated_random_piece()
                if self.render:
                    self.render_frame(np.copy(self.board))
            # Else add a checkpoint attempt and load checkpoint if limits have been reached
            if not self.carve(random_piece, rotations, location, len(self.random_piece_generator) == 7):
                if checkpoint_manager.add_attempt():
                    checkpoint, reverted = checkpoint_manager.load_checkpoint()
                    self.board = np.copy(checkpoint)
                    self.pieces = self.pieces[:len(self.random_piece_generator) - (
                        14 if reverted else 7)]
                    if self.debug:
                        self.solution = self.solution[:len(self.random_piece_generator) - (
                            14 if reverted else 7)]
                    self.random_piece_generator.generate_pieces()

        # If less pieces were used than the allows maximum then pad out the pieces randomly
        if (len(self.pieces) <= self.M):
            padding = self.random_piece_generator.get_random_sequence(
                self.M - len(self.pieces) + 1)
            padding.reverse()
            self.pieces[:0] = padding

    def carve(self, piece: int, rotations: int, location: int, allow_partial: bool) -> bool:
        # Unpack information
        tetromino, reverse_tetromino_topography = get_tetromino(
            piece, rotations)
        tetromino_height, tetromino_width = tetromino.shape

        # Calculate drop location
        drop_deltas = self.calculate_drop_deltas(
            location, reverse_tetromino_topography, tetromino_width)
        drop = self.calculate_drop(drop_deltas)

        # Calculate the push needed to fully immerse the piece in the board
        push = reverse_tetromino_topography[np.argmin(drop_deltas)] + 1

        # Add the push to the drop
        drop += push

        # If partial carve is allowed then try every possible different push, else just try the maximum push
        increments = tetromino_height if allow_partial else 1
        for increment in range(increments):
            if self.calculate_carve(drop, location, tetromino, reverse_tetromino_topography, allow_partial):
                return True
            drop -= 1

        # If no carving was possible then carving failed
        return False

    def calculate_carve(self, drop: int, location: int, tetromino: np.array, reverse_tetromino_topography: tuple[int, ...], allow_partial: bool):
        tetromino_height, tetromino_width = tetromino.shape

        # If the drop goes out of bounds then the carve failed
        if drop + tetromino_height > 20:
            return False

        # If partial carving is disallowed then check for overlap
        if not allow_partial:
            # Calculate overlap
            board_slice = self.board[drop:drop + tetromino_height,
                                     location:location + tetromino_width]
            overlap = np.all(np.logical_not(tetromino) | board_slice)

            # If no overlap then carving failed
            if not overlap:
                return False

        # The carving operation as defined is not reversible, so we save the current slice to be able to invert if needed
        board_checkpoint = np.copy(
            self.board[drop:drop + tetromino_height, location:location + tetromino_width])

        # Apply the carve
        self.board[drop:drop + tetromino_height, location:location +
                   tetromino_width] &= np.logical_not(tetromino)

        # Make a new drop to ensure it falls where the carve was
        # Ensures that there were no blocking blocks above and that there were supporting blocks below
        new_drop_deltas = self.calculate_drop_deltas(
            location, reverse_tetromino_topography, tetromino_width)
        new_drop = self.calculate_drop(new_drop_deltas)

        # If the new drop does not land where the carve was done then revert the carve and carving failed
        if new_drop != drop:
            self.board[drop:drop + tetromino_height,
                       location:location + tetromino_width] = board_checkpoint
            return False

        # If nothing faield the carve then it succeeded
        return True

    def move(self, rotations: int, location: int) -> None:
        # Get the next piece
        piece = self.pieces.pop()

        # Get the tetromino and unpack it's information
        tetromino, reverse_tetromino_topography = get_tetromino(
            piece, rotations)

        # Ensure the location is not out of bounds horizontally
        tetromino_width = tetromino.shape[1]
        location = min(location, 10-tetromino_width)

        # Calculate the drop height
        drop_deltas = self.calculate_drop_deltas(
            location, reverse_tetromino_topography, tetromino_width)
        drop = self.calculate_drop(drop_deltas)

        # Check if out of bounds
        if drop < 0:
            self.state = False
            return

        # Insert Block
        self.board[drop:drop + tetromino.shape[0],
                   location:location + tetromino_width] |= tetromino
        self.moves_used += 1

        # Clear lines
        rows_all_trues = np.all(
            self.board[drop:drop + tetromino.shape[0], :], axis=1)

        # Count the number of lines that were cleared
        rows_cleared = np.count_nonzero(rows_all_trues)

        # If no lines cleared
        if rows_cleared == 0:
            if self.moves_used >= self.M:
                self.state = False
            if self.render:
                self.render_frame(np.copy(self.board))
            return

        # Collect the indices of all lines that need to be cleared
        indices_to_clear = []
        for row_index, row in enumerate(rows_all_trues):
            if row:
                indices_to_clear.append(drop + row_index)

        indices_left = [x for x in range(0, 20) if x not in indices_to_clear]

        # Apply the line clears
        self.board = self.board[indices_left]
        rows_to_add = np.full((rows_cleared, 10), False)
        self.board = np.vstack((rows_to_add, self.board))

        self.lines_cleared += rows_cleared

        if self.render:
            self.render_frame(np.copy(self.board))

        # Check for win condition
        if self.lines_cleared >= self.L:
            self.state = True
            return

        # Check for loose condition
        if self.moves_used >= self.M:
            self.state = False
            return

    def calculate_drop(self, drop_deltas) -> int:
        return min(drop_deltas) - 1

    def calculate_drop_deltas(self, location, reverse_tetromino_topography, tetromino_width):
        board_topography = []
        for col in self.board.T[location:location+tetromino_width, :]:
            result = np.where(col)[0]
            board_topography.append(result[0] if len(result) != 0 else 20)

        return np.array(board_topography) - np.array(reverse_tetromino_topography)

    def get_state(self) -> tuple[np.array, int, int, int, int, bool]:
        return self.board, self.pieces[-1], self.pieces[-2], self.L - self.lines_cleared, self.M - self.moves_used, self.state

    def reset(self) -> None:
        self.random_piece_generator.reset()
        self.board[:, :] = False
        self.pieces.clear()

        self.load_warm_reset()

    def load_warm_reset(self) -> None:
        if self.warm_reset:
            self.board, self.pieces = self.queue.get()
        else:
            self._generate_initial_config()

    def terminate(self):
        # If using warm reset then we need to ensure we close all processes
        if self.warm_reset:
            # Set the terminate event to terminate processes
            self.terminate_event.set()

            # Clear the queue to ensure processes are not prevented from terminating due to being stuck on a full queue
            while not self.queue.empty():
                self.queue.get()

            # Close the queue
            self.queue.close()

            # Wait for processes to finish
            self.warm_reset_process.join()
            self.forward_warm_reset_process.join()

        # If using render then quit pygame
        if self.render:
            self.pygame.quit()


def warm_reset_worker(queue: multiprocessing.Queue, warm_reset_tetris, terminate_event: multiprocessing.Event):
    while not terminate_event.is_set():
        warm_reset_tetris.reset()
        reset_point = (np.copy(warm_reset_tetris.board),
                       warm_reset_tetris.pieces.copy())

        queue.put(reset_point)


def forward_warm_reset_worker(queue: multiprocessing.Queue, terminate_event: multiprocessing.Event, L, M):
    while not terminate_event.is_set():
        batch = translate(main.generate_batch(L, M, debug=False))
        for reset_point in batch:
            if terminate_event.is_set():
                break
            queue.put(reset_point)

import unittest
import tetris

class TestTetris(unittest.TestCase):
    def test_full(self):
        game = tetris.Tetris(15, 40, debug=True)
        for move in game.solution[::-1]:
            game.move(move[0], move[1])
        self.assertTrue(game.state)
        game.terminate()

if __name__ == '__main__':
    game = tetris.Tetris(3, 5, warm_reset=False, debug=True)
    print(game.board, game.pieces)
    for move in game.solution[::-1]:
        game.move(move[0], move[1])
        print(game.board, move, game.state)
    game.terminate()

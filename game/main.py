import unittest
import tetris
import numpy as np

class TestRandomPieceGeneration(unittest.TestCase):
    def test_regenerate(self):
        random_piece_generator = tetris.RandomPieceGenerator()

        for i in range(16):
            (random_piece, random_piece_index), regenerated = random_piece_generator.get_random_piece()
            self.assertEqual(regenerated, i % 7 == 0, 'Regeneration notification failed')

            self.assertEqual(len(random_piece_generator), 7 - (i % 7), 'Regeneration failed')

            random_piece_generator.delete_index(random_piece_index)

            self.assertEqual(len(random_piece_generator), 7 - (i % 7) - 1, 'Deletion failed')

    def test_sequence(self):
        random_piece_generator = tetris.RandomPieceGenerator()
        sequence_length = 16
        sequence = random_piece_generator.get_random_sequence(sequence_length)
        
        self.assertEqual(len(sequence), sequence_length, 'Sequence of wrong length')

        for i in range(0, len(sequence), 7):
            permutation_group = sequence[i:i+7]
            self.assertEqual(len(permutation_group), len(set(permutation_group)), 'Groups contain duplicates')

class TestTetris(unittest.TestCase):
    def test_carving_repeatability(self):
        L, M = 15, 40
        game = tetris.Tetris(L, M, warm_reset=False, debug=True)

        comparative_game = tetris.Tetris(L, M, warm_reset=False, debug=True)
        comparative_game.board[-L:, :] = True


        for i in range(len(game.solution) - 1, -1, -1):
            piece = game.pieces[i]
            rotations, location = game.solution[i]
            self.assertTrue(comparative_game.carve(piece, rotations, location, i==(len(game.solution) - 1)), 'Carving with solution failed')

        game.terminate()

        self.assertTrue(np.array_equal(game.board, comparative_game.board), 'Carving repeat failed')

    def test_carving_invertability(self):
        L, M = 15, 40

        game = tetris.Tetris(L, M, warm_reset=False, debug=True)

        for i in range(0, len(game.solution)):
            rotations, location = game.solution[i]
            game.move(rotations, location)
        self.assertTrue(game.state, 'Carving inversion failed')

if __name__ == '__main__':
    unittest.main()

import tetris
import time

if __name__ == '__main__':
    timer = time.time()
    game = tetris.Tetris(15, 40, warm_reset=True)
    for _ in range(10):
        game.reset()
        print(game.board, game.pieces, len(game.pieces))
    game.terminate()
    timer = time.time() - timer
    print(timer)

import tetris
import time

if __name__ == '__main__':
    timer = time.time()
    game = tetris.Tetris(15, 40, warm_reset=True)
    print(game.board)
    for _ in range(10):
        game.reset()
        print(game.board)
        time.sleep(1)
    game.reset()
    print(time.time() - timer)
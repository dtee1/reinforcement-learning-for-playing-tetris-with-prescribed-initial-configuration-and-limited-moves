from tetris import Tetris, tetrominos
from random import randint
from time import time

#print(tetrominos)
#'''
game = Tetris(15, 40, random_pieces=True)

MOVES = 100000

resets = 0
start_time = time()
for _ in range(MOVES):
    game.move(randint(0, 3), randint(0, 10))
    if game.state != None:
        game.reset()
        resets += 1
end_time = time()
elapsed_time = end_time - start_time
moves_per_second = round(MOVES / elapsed_time)

print(f'Played {MOVES} moves across {resets} games with an average of {moves_per_second} moves per second')#'''
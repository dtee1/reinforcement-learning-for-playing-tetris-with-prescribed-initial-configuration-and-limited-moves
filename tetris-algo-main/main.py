from TetrisSolver import TetrisSolver
from TetrisGameGenerator import TetrisGameGenerator
from time import time
import multiprocessing

def solve_game(game):
    solver = TetrisSolver(game.board, game.sequence, game.goal, 10000)
    result, moves, failed_attempts = solver.solve()
    if result:
        return game
    else:
        return None

if __name__ == "__main__":
    games = []
    winnable_games = []

    for i in range(10000):
        games.append(TetrisGameGenerator(seed=i, goal=10, tetrominoes=50))

    # Number of processes to run in parallel
    num_processes = multiprocessing.cpu_count()

    start_loop = time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        winnable_games = pool.map(solve_game, games)

    # Remove None values from the list
    winnable_games = [game for game in winnable_games if game is not None]

    end_loop = time()
    print("Total time: ", end_loop - start_loop)
    print("Number of winnable games: ", len(winnable_games))

    # Create a CSV file with the winnable games and their seed | max_moves | goal | initial_height_max
    import csv

    with open('winnable_games.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "max_moves", "goal", "initial_height_max"])
        for game in winnable_games:
            writer.writerow([game.seed, game.tetrominoes, game.goal, game.initial_height_max])

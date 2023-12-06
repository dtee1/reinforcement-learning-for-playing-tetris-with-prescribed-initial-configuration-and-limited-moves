from TetrisSolver import TetrisSolver
from TetrisGameGenerator import TetrisGameGenerator
from time import time
import multiprocessing
import csv

def solve_game(args):
    game, max_moves, test = args
    solver = TetrisSolver(game.board, game.sequence, game.goal, max_attempts=max_moves)

    result, moves, failed_attempts = solver.solve()
    if(test):
        return {
                "solvable": result,
                "failed_attempts": failed_attempts
                }


    return game if result else None

def generate_game(args):
    seed, goal, tetrominoes, initial_height_max = args
    game = TetrisGameGenerator(seed=seed, goal=goal, tetrominoes=tetrominoes,initial_height_max= initial_height_max)
    return game

if __name__ == "__main__":
    winnable_games = []
    attempts = []
    games = []

    num_processes = multiprocessing.cpu_count()
    print(f"Number of processes: {num_processes}")

    # MODIFIABLE PARAMETERS
    goal = 8
    tetrominoes = 40
    initial_height_max = 4
    start = 0
    end = 100
    max_attempts = 1000
    # =====================

    start_loop = time()

    # start_minimization = time()

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     games = pool.map(generate_game, [(i, goal, tetrominoes, initial_height_max) for i in range(0, test_games_to_generate)])

    # start_minimization = time()

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     attempts = pool.map(solve_game, [(game, max_attempts, True) for game in games])

    # max_attempts = minimize_max_attempts(attempts)
    # print(f"Best max_attempts: {max_attempts}")
    # print(f"Time to minimize max_attempts: {time() - start_minimization}")


    start_game_generation = time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        games += pool.map(generate_game, [(i, goal, tetrominoes, initial_height_max) for i in range(start, end)])

    end_game_generation = time()
    print(f"Time to generate games: {end_game_generation - start_game_generation}")


    start_game_solving = time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        winnable_games = pool.map(solve_game, [(game, max_attempts, False) for game in games])
    winnable_games = [game for game in winnable_games if game is not None]
    end_game_solving = time()
    print(f"Time to solve games: {end_game_solving - start_game_solving}")

    end_loop = time()

    print("Total time: ", end_loop - start_loop)
    print("Number of winnable games: ", len(winnable_games))

    with open('log.txt', 'a') as file:
        file.write(f"The average time per winnable game for {goal}/{tetrominoes} goal/tetrominoes was {(end_loop - start_loop) / len(winnable_games)} seconds. {len(winnable_games)} games were winnable. It took {end_loop - start_loop} seconds to pass through all {len(games)} seeds. with a max_attempts of {max_attempts}.\n")

    # Create a CSV file with the winnable games and their seed | max_moves | goal | initial_height_max
    if(len(winnable_games) > 0):
        with open('winnable_games.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["seed", "max_moves", "goal", "initial_height_max"])
            for game in winnable_games:
                writer.writerow([game.seed, game.tetrominoes, game.goal, game.initial_height_max])

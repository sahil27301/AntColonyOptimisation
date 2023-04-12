import argparse
import logging
from argparse import Namespace

from grid import Grid


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%H:%M:%S")


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def closest_food_distance(row: int, column: int, food_sources: list[tuple[int, int]]) -> int:
    return min(manhattan_distance(row, column, x, y) for (x, y) in food_sources)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        prog='AntColonyOptimisation',
        description='Creates an ant colony simulation',
    )
    parser.add_argument('--rows', type=int, default=100)
    parser.add_argument('--columns', type=int, default=100)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num-ants', type=int, default=500)
    parser.add_argument('--num-food', type=int, default=10)
    parser.add_argument('--min-pheromone', type=float, default=1)
    parser.add_argument('--max-pheromone', type=float, default=100)
    parser.add_argument('--evaporation-rate', type=float, default=0.1)
    args: Namespace = parser.parse_args()
    rows: int = args.rows
    columns: int = args.columns
    density: float = args.density
    num_ants: int = args.num_ants
    num_food: int = args.num_food
    min_pheromone: float = args.min_pheromone
    max_pheromone: float = args.max_pheromone
    evaporation_rate: float = args.evaporation_rate
    grid: Grid = Grid(
        rows,
        columns,
        density,
        num_ants,
        num_food,
        min_pheromone,
        max_pheromone,
        evaporation_rate,
    )

    grid.generate()

    grid.draw()


if __name__ == '__main__':
    main()

import random

import numpy as np


def manhattan_distance(x1: int, y1: int, x2: int, y2: int):
    return abs(x1 - x2) + abs(y1 - y2)


def get_heuristic(row: int, column: int, food: set[tuple[int, int]]):
    return min(manhattan_distance(row, column, x, y) for (x, y) in food)


def choose_next_cell(neighbours: list[tuple[int, int]], probabilities: list[float]) -> tuple[int, int]:
    total_probability: float = sum(probabilities)
    partition: float = random.uniform(0, total_probability)
    running_total: float = 0
    for neighbour, probability in zip(neighbours, probabilities):
        running_total += probability
        if running_total >= partition:
            return neighbour

    # In case of some rounding error causing no neighbour to be selected
    return neighbours[-1]


class Ant:
    def __init__(self, row: int, column: int, max_pheromone: float) -> None:
        self.row = row
        self.column = column
        self.max_pheromone = max_pheromone
        self.visited_cells: list[tuple[int, int]] = [(row, column)]
        self.found_food: bool = False
        self.pheromone_strength: float = 0
        self.visit_count: dict[tuple[int, int], int] = {}

    def move(self, neighbours: list[tuple[int, int]], food: set[tuple[int, int]], pheromones: np.ndarray):
        if not self.found_food:
            self.explore(neighbours, food, pheromones)
        else:
            self.backtrack(pheromones)

    def get_probability(self, pheromone: float, heuristic: int, neighbour: tuple[int, int]):
        alpha: float = 0.5  # Controls how strongly ants follow the pheromone trail
        beta: float = 0.5  # Controls how strongly ants follow the heuristic
        gamma: float = 2  # Controls how strongly ants avoid previously visited cells
        return pow(pheromone, alpha) * pow(1 / max(1, heuristic), beta) * pow(1 / (1 + self.visit_count.get(neighbour, 0)), gamma)

    def explore(self, neighbours: list[tuple[int, int]], food: set[tuple[int, int]], pheromones: np.ndarray):
        if len(neighbours) == 0:
            return
        heuristics = [get_heuristic(row, column, food) for (row, column) in neighbours]
        transition_probabilities: list[float] = []
        neighbour_pheromones: list[float] = [pheromones[row, column] for (row, column) in neighbours]
        for pheromone, heuristic, neighbour in zip(neighbour_pheromones, heuristics, neighbours):
            transition_probabilities.append(self.get_probability(pheromone, heuristic, neighbour))
        total_probability = sum(transition_probabilities)
        for index, probability in enumerate(transition_probabilities):
            transition_probabilities[index] = probability / total_probability

        self.row, self.column = choose_next_cell(neighbours, transition_probabilities)

        self.visited_cells.append((self.row, self.column))

        self.visit_count[(self.row, self.column)] = self.visit_count.get((self.row, self.column), 0) + 1

        if (self.row, self.column) in food:
            self.found_food = True
            self.pheromone_strength = self.max_pheromone / len(self.visited_cells)

    def backtrack(self, pheromones: np.ndarray):
        if len(self.visited_cells) == 0:
            self.found_food = False
            self.visit_count.clear()
            return
        self.row, self.column = self.visited_cells.pop()
        pheromones[self.row, self.column] += self.pheromone_strength

    def __repr__(self) -> str:
        return f"Ant[{self.row}, {self.column}]"

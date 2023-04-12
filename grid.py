import logging
from random import sample

import numpy as np
import pygame

from ant import Ant


def get_neighbours(row: int, column: int) -> list[tuple[int, int]]:
    answer: list[tuple[int, int]] = []
    for delta_x in (0, 1, -1):
        for delta_y in (0, 1, -1):
            if delta_x == delta_y == 0:
                continue
            answer.append((row + delta_x, column + delta_y))
    return answer


class Grid:
    """
    Grid object that contains the map, ants, food sources, and pheromone trails
    """

    def __init__(
            self,
            rows: int,
            columns: int,
            density: float = 0.2,
            num_ants: int = 500,
            num_food: int = 10,
            min_pheromone: float = 1,
            max_pheromone: float = 100,
            evaporation_rate: float = 0.1,
    ) -> None:
        """
        :param rows: The number of rows in the grid
        :param columns: The number of columns in the grid
        :param density: The density of obstacles in the grid. Should lie in [0, 0.8]
        :param num_ants: the number of ants that will be generated
        :param num_food: the number of food that will be generated
        :param min_pheromone: The lowest possible pheromone value in a cell
        :param max_pheromone: The highest possible pheromone value in a cell
        """
        self.rows = rows
        self.columns = columns
        self.density = max(min(density, 0.8), 0)
        if density < 0 or density > 0.8:
            logging.warning(f"Density must be in the range [0, 0.8]. Using {self.density}")
        self.num_ants = num_ants
        self.num_food = num_food
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.evaporation_rate = evaporation_rate
        self.grid = np.zeros(shape=[rows, columns])
        self.ants: list[Ant] = []
        self.food: set[tuple[int, int]] = set()
        self.pheromones: np.ndarray = np.ones_like(self.grid)

    def generate(self) -> None:
        """
        Generates the obstacles, ants and pheromone trails
        :return:
        """
        num_obstacles: int = round(self.rows * self.columns * self.density)
        cells: list[tuple[int, int]] = [
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
        ]
        for (row, column) in sample(cells, num_obstacles):
            self.grid[row, column] = 1

        empty_cells: list[tuple[int, int]] = [(row, column)
                                              for row in range(self.rows)
                                              for column in range(self.columns)
                                              if self.grid[row, column] == 0]

        # self.ants = [Ant(row, column, self.max_pheromone) for (row, column) in sample(empty_cells, self.num_ants)]
        self.ants = [Ant(empty_cells[0][0], empty_cells[0][1], self.max_pheromone) for _ in range(self.num_ants)]

        self.food = sample(empty_cells, self.num_food)

        for (row, column) in self.food:
            self.pheromones[row, column] = 8 * self.max_pheromone / 10
            for (x, y) in get_neighbours(row, column):
                if self.is_valid(x, y):
                    self.pheromones[x, y] = 6 * self.max_pheromone / 10

    def is_valid(self, row, column) -> bool:
        return 0 <= row < self.rows and 0 <= column < self.columns

    def update(self):
        for ant in self.ants:
            neighbours: list[tuple[int, int]] = [
                neighbour
                for neighbour in get_neighbours(ant.row, ant.column)
                if self.is_valid(neighbour[0], neighbour[1])
            ]
            ant.move(neighbours, self.food, self.pheromones)

        for row in range(self.rows):
            for column in range(self.columns):
                self.pheromones[row, column] -= self.evaporation_rate
                self.pheromones[row, column] = min(self.max_pheromone, max(self.min_pheromone, self.pheromones[row, column]))


    def draw(self) -> None:
        pygame.init()
        cell_size: int = 10
        width, height = cell_size * self.columns, cell_size * self.rows
        screen: pygame.Surface = pygame.display.set_mode((width, height))
        white: tuple[int, int, int] = (255, 255, 255)  # Empty cells
        black: tuple[int, int, int] = (0, 0, 0)  # Obstacles
        red: tuple[int, int, int] = (255, 0, 0)  # Ants
        green: tuple[int, int, int] = (0, 255, 0)  # Food

        running: bool = True
        clock: pygame.time.Clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update()

            visualisation_grid: np.ndarray = self.get_visualisation_grid()

            for row in range(self.rows):
                for column in range(self.columns):
                    color = None
                    cell_value: int = visualisation_grid[row, column]
                    pheromone_value: float = self.pheromones[row, column]
                    normalised_pheromone_value: float = (pheromone_value - self.min_pheromone) / (
                            self.max_pheromone - self.min_pheromone)

                    if cell_value == 0:
                        color = white
                    elif cell_value == 1:
                        color = black
                    elif cell_value == 2:
                        color = red
                    elif cell_value == 3:
                        color = green
                    pygame.draw.rect(screen, color, (column * cell_size, row * cell_size, cell_size, cell_size))

                    if normalised_pheromone_value > 0 and cell_value == 0:
                        alpha = round(normalised_pheromone_value * 255)
                        pheromone_surface: pygame.Surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                        pheromone_surface.fill(pygame.Color(0, 0, 255, alpha))
                        screen.blit(pheromone_surface, (column * cell_size, row * cell_size))

            pygame.display.flip()

            clock.tick(100)

        pygame.quit()

    def get_visualisation_grid(self) -> np.ndarray:
        visualisation_grid: np.ndarray = self.grid.copy()
        for ant in self.ants:
            visualisation_grid[ant.row, ant.column] = 2
        for (row, column) in self.food:
            visualisation_grid[row, column] = 3
        return visualisation_grid

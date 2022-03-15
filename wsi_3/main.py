"""
Introduction to Artificial Intelligence, Exercise 3:
Two-person deterministic games.
Author: Robert Kaczmarski 293377
"""
import pygame
import argparse
from typing import Optional, Sequence

from game.gameplay import Gameplay
from game.locals import WIDTH, HEIGTH


# WINDOW = pygame.display.set_mode((WIDTH, HEIGTH))
# pygame.display.set_caption("Fox and Hounds")


def main(argv: Optional[Sequence[int]] = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--depth")
    parser.add_argument("-fsc", "--fox_starting_column")
    parser.add_argument("-N", "--number_of_games")
    args = parser.parse_args(argv)

    FPS = 60
    clock = pygame.time.Clock()
    depth = int(args.depth)
    fox_starting_column = int(args.fox_starting_column)
    number_of_games = int(args.number_of_games)

    for i in range(0, number_of_games):
        WINDOW = pygame.display.set_mode((WIDTH, HEIGTH))
        gameplay = Gameplay(WINDOW, fox_starting_column)
        pygame.display.set_caption("Fox and Hounds")
        gameplay.game_loop(clock, FPS, depth, i)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

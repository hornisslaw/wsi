from __future__ import annotations

import matplotlib.pyplot as plt


def split_list(
        list_of_lists: list[list[float, float]]
) -> tuple[list[float], list[float]]:
    return map(list, zip(*list_of_lists))


def plot_cities(positions: list[list[float, float]]) -> None:
    xs, ys = split_list(positions)
    # print(f'xs = {xs}')
    # print(f'ys = {ys}')
    plt.plot(xs, ys, "ro--")
    # plt.plot(xs, ys, "ro")
    plt.plot(xs[0], ys[0], "y*--")
    plt.plot(xs[-1], ys[-1], "y*--")
    plt.xlabel("X")
    plt.ylabel("Y")


def plot_cycle(
        t: int, 
        min_cycle_distance: float, 
        minimal_chromosome: list[list[float, float]]
) -> None: 
    plt.clf()
    plt.title(f"Generation: {t}, Minimal cycle: {min_cycle_distance:.2f}")
    plot_cities(minimal_chromosome)
    plt.pause(0.001)
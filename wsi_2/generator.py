from __future__ import annotations

import numpy as np

from sklearn.datasets import make_blobs

from plots import plot_cities


def generate_cities_with_chessboard_distribution(city_names: list[str]) -> dict:
    cities = {}
    max_x = 1000
    max_y = 1000
    rows = 5
    columns = 6
    positions = []
    for i in range(0, columns):
        for j in range(0, rows):
            positions.append([j*max_x/(rows-1), i*max_y/(columns-1)])
    for name, position in zip(city_names, positions):
        cities[name] = np.array(position)
    
    # plot_cities(positions)
    return cities


def generate_cities_with_large_centres_distribution(
        city_names: list[str]
) -> dict:
    cities = {}
    positions, _ = make_blobs(n_samples=30, 
                              centers=2, 
                              cluster_std = 20, 
                              center_box = (0, 1000))
    for name, position in zip(city_names, positions):
        cities[name] = position
    
    # plot_cities(positions)
    return cities


def generate_cities_randomly(city_names: list[str]) -> dict:
    cities = {}
    N = 30
    positions = 1000*np.random.random((N,2))
    for name, position in zip(city_names, positions):
        cities[name] = position
    # plot_cities(positions)
    return cities


def generate_cities(distribution: str, city_names: list) -> dict:
    """Create 1000x1000 grid and select [x,y] coordinates to generate 
    cities positions.
    Rozkłady miast: jednorodny, duże skupiska, losowy.
    """
    cities = {}
    dist = distribution.lower().strip()
    if dist == "chessboard":
        cities = generate_cities_with_chessboard_distribution(city_names)
    elif dist == "large_centres":
        cities = generate_cities_with_large_centres_distribution(city_names)
    elif dist == "random":
        cities = generate_cities_randomly(city_names)
    else:
        print(f"Only chessboard/large_centres/random, given: {distribution}")
    return cities
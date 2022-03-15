"""
Introduction to Artificial Intelligence, Exercise 2:
Genetic and evolutionary algorithms.
Author: Robert Kaczmarski 293377
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import random
import time

from generator import generate_cities
from algorithm import genetic_algorithm


def measure_performence(max_generations: int, 
                        population_size: int, 
                        cities: dict, 
                        starting_city: str, 
                        mutation_probability: float,
                        K: int,
                        plot: bool
) -> tuple[float, list[float], float]:
    time_start = time.monotonic_ns()
    minimum, visited_minimas = genetic_algorithm(max_generations, 
                                                population_size, 
                                                cities, 
                                                starting_city, 
                                                mutation_probability, 
                                                K,
                                                plot)
    time_stop = time.monotonic_ns()
    total_time = time_stop - time_start
    return minimum, visited_minimas, total_time


def load_city_names(filename: str) -> list[str]:
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines


def main() -> int:
    distributions = ["chessboard", "large_centres", "random"]
    city_names = load_city_names("cities.txt")

    np.random.seed(0)
    population_size = 50
    max_generations = 1000
    mutation_probability = 0.1
    K = 2
    plot = False
    starting_city = "Ketowice"
    distr = distributions[2]
    cities = generate_cities(distr, city_names)

    list_of_minimum = []
    list_of_minimas = []
    list_of_total_times = [] 

    for i in range (1, 11):
        # plt.figure(i)
        random.seed(i)
        minimum, minimas, total_time = measure_performence(max_generations, 
                                                           population_size, 
                                                           cities, 
                                                           starting_city, 
                                                           mutation_probability, 
                                                           K,
                                                           plot)
        list_of_minimum.append(minimum)
        list_of_minimas.append(minimas)
        list_of_total_times.append(total_time)


    print(f"Best: {min(list_of_minimum)}")
    print(f"Avg: {np.mean(np.array(list_of_minimum))}")
    print(f"Std: {np.std(np.array(list_of_minimum))}")
    print(f"Avg time: {np.mean(np.array(list_of_total_times))*1e-9}s")
    
    # plt.title("Najkrótsze znalezione cykle w kolejnych generacjach")
    # plt.plot(list_of_minimas[0])
    # plt.xlabel("Generacja")
    # plt.ylabel("Najkrótszy cykl")
    # plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
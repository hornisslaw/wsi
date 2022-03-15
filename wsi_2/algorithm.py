from __future__ import annotations

import numpy as np
import random
import matplotlib.pyplot as plt

from plots import plot_cycle


def initial_population(
        cities: dict, 
        population_size: int, 
        starting_city: str
) -> list[list[str]]:
    population = []
    city_names = list(cities.keys())
    city_names.remove(starting_city)
    for i in range(0, population_size):
        rand_seq = random.sample(city_names, len(city_names))
        rand_seq.insert(0, starting_city)
        rand_seq.append(starting_city)
        population.append(rand_seq)
    return population


def calculate_distance(
        city_1: list[float, float], 
        city_2: list[float, float]
) -> float:
    return np.sqrt(np.add(np.power(np.subtract(city_1[0], 
                                               city_2[0]), 2), 
                          np.power(np.subtract(city_1[1], 
                                               city_2[1]), 2)))


def calculate_cycle_distance(cities : dict, chromosome: list[float]) -> float:
    total_distance = 0
    for i in range(1, len(chromosome)):
        total_distance += calculate_distance(cities[chromosome[i-1]], 
                                             cities[chromosome[i]])
    return total_distance


def tournament_selection(
        cities: dict, 
        population: list[list[str]], 
        K: int=2
)-> list[str]:
    winners = []
    for i in range(0, len(population)):
        competitors = random.sample(population, K)
        distances = [calculate_cycle_distance(cities, c) for c in competitors]
        index_of_minimal = np.argmin(distances)
        winners.append(competitors[index_of_minimal].copy())
    return winners


def mutation(
        population: list[list[str]], 
        mutation_probability: float=0.01
) -> list[list[str]]:
    for popul in population:
        if random.random() <= mutation_probability:
            first, second = random.sample(range(1, len(popul)-1), 2)
            popul[first], popul[second] = popul[second], popul[first]
    return population    


def check_minimal_cycle(
        cities: dict, 
        population: list[list[str]], 
        previous_minimal_cycle_distance: float=np.Inf
    ) -> tuple[float, list[float]]:
    new_minimal_cycle_distance = np.Inf
    chromosome = []
    for p in population:
        cycle_distance = calculate_cycle_distance(cities, p)
        if cycle_distance < previous_minimal_cycle_distance:
            new_minimal_cycle_distance = cycle_distance
            chromosome = [cities[name] for name in p]
        else:
            pass
    return new_minimal_cycle_distance, chromosome


def genetic_algorithm(
        max_generations: int, 
        population_size: int, 
        cities: dict, 
        starting_city: str, 
        mutation_probability: float,
        K: int,
        plot: bool
) -> tuple[float, list[float]]:

    P_t = initial_population(cities, population_size, starting_city)
    t = 0
    min_cycle_distance_found, min_chromosome = check_minimal_cycle(cities, P_t)
    history_of_minimals = [min_cycle_distance_found]
    while t < max_generations:
        print(f"Generation: {t}, Minimal: {min_cycle_distance_found}")
        T_t = tournament_selection(cities, P_t)
        O_t = mutation(T_t, mutation_probability)
       
        possible_minimum, possible_chromosome = check_minimal_cycle(
                                            cities, 
                                            O_t, 
                                            min_cycle_distance_found)
        if possible_minimum < min_cycle_distance_found:
            min_cycle_distance_found = possible_minimum
            min_chromosome = possible_chromosome
        P_t = O_t
        t += 1
        history_of_minimals.append(min_cycle_distance_found)

        # if plot:
        #     plot_cycle(t, min_cycle_distance_found, min_chromosome)
    if plot:
        plot_cycle(t, min_cycle_distance_found, min_chromosome)

    return min_cycle_distance_found, history_of_minimals

    # cities {name, x, y}
    # popul = list[list[dict]]
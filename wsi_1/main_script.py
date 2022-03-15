"""
Introduction to Artificial Intelligence, Exercise 1:
Comparison of the gradient descent method and the Newton method.
Author: Robert Kaczmarski 293377
"""
import matplotlib.pyplot as plt
import numpy as np
import time

from methods import find_minimum, gradient_search_direction, himmelblau, newton_search_direction


def measure_performence(starting_points, beta, max_iters, search_direction):
    time_start = time.monotonic_ns()
    minimum, visited_points = find_minimum(starting_points, beta, max_iters, search_direction)
    time_stop = time.monotonic_ns()
    total_time = time_stop - time_start
    return minimum, visited_points, total_time

def plot_contour3d(x, y, F, contours):
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, F, contours)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Himmelblau Function')
    ax.view_init(50, 50)

def plot_path(x_points, y_points, label_name, color: str):
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(X, Y)
    F = himmelblau([x, y])
    contours = 50

    plt.contour(x, y, F, contours)
    plt.plot(x_points, y_points, color+"o", label = label_name, linewidth=1, markersize=2, linestyle='dashed')
    # Plot the starting point
    plt.plot(x_points[0], y_points[0], "yp", markersize=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend(bbox_to_anchor=(0, 1.02, 1,0), loc=3, ncol=1, borderaxespad=0)

def display_results(method_name, minimum, visited_points, total_time):
    nano = 1e-9
    print(f'Results for {method_name}:')
    print(f'Minimum found at {minimum}, number of points = {len(visited_points)}')
    print(f'Performence time: {total_time*nano:.15f} sec' )

def split_list(list_of_lists):
    return map(list, zip(*list_of_lists))

def main() -> int: 
    starting_points = [-4.0, 0.0]
    beta = 0.001
    max_iters = 100000

    minimum_grad, visited_points_grad, time_grad = measure_performence(starting_points, beta, max_iters, gradient_search_direction)
    display_results("Gradient", minimum_grad, visited_points_grad, time_grad)
    minimum_newt, visited_points_newt, time_newt = measure_performence(starting_points, beta, max_iters, newton_search_direction)
    display_results("Newton", minimum_newt, visited_points_newt, time_newt)

    # visited_points_grad = [[x0,y0], [x1,y1], ...]
    # Splitting to two lists
    x_grad, y_grad = split_list(visited_points_grad)
    x_newt, y_newt = split_list(visited_points_newt)

    plot_path(x_grad, y_grad, 'Metoda spadku gradientu', "b")
    plot_path(x_newt, y_newt, 'Metoda Newtona', "r")
    plt.show()
    # plot_contour3d(x, y, F, 80)
    # plt.plot([3.0, -2.805118, -3.779310, 3.584428], [2.0, 3.131312, -3.283186, -1.848126], 'ro')
    # plt.show()
    
    return 0

if __name__ == "__main__":
    raise(SystemExit(main()))
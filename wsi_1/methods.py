"""
Implementation of the steepest gradient descent and Newton 
method.
"""
from __future__ import annotations

import numpy as np

from numpy import Inf
from numpy.linalg import inv

# Himmelblau's function
f = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
# Gradient
df_dx = lambda x: 2*(x[0]**2+x[1]-11)*2*x[0] + 2*(x[0]+x[1]**2-7)
df_dy = lambda x: 2*(x[0]**2+x[1]-11) + 2*(x[0]+x[1]**2-7)*2*x[1]
# Hessian
d2f_dx2 = lambda x: 8*x[0]**2 + 4*(x[0]**2+x[1]-11) + 2
d2f_dxy = lambda x: 4*x[0] + 4*x[1]
d2f_dyx = lambda x: 4*x[0] + 4*x[1]
d2f_dy2 = lambda x: 2 + 8*x[1]**2 + 4*(x[0]+x[1]**2-7)

def himmelblau(params: list[float]) -> float:
    return (params[0]**2+params[1]-11)**2 + (params[0]+params[1]**2-7)**2

def gradient_q(params: list[float]):
    return np.array([df_dx(params), df_dy(params)])

def hessian_q(params: list[float]):
    return np.array(
            [[d2f_dx2(params), d2f_dxy(params)],
            [d2f_dyx(params), d2f_dy2(params)]]
            )

def newton_search_direction(x: np.ndarray) -> np.ndarray:
    return np.dot(inv(hessian_q(x)), gradient_q(x))

def gradient_search_direction(x: np.ndarray) -> np.ndarray:
    return gradient_q(x)

def find_minimum(
    starting_pos: list[float], 
    beta: float, 
    max_iters: float,
    d: function, 
    epsilon: float=10e-12
    ) -> list[float]:

    x = np.array(starting_pos, dtype='double')
    step = len(starting_pos)*[Inf]
    iter = 0
    visited_points = [x.copy()]
    while iter < max_iters and not np.allclose(step, np.zeros(len(step)), atol=epsilon):
        step = beta*d(x)
        x -= step
        # print(f"Iter: {iter}, Step: {step}, x = {x}")
        iter += 1
        visited_points.append(x.copy())
    return x, visited_points

def main() -> int:
    X = [0., 0]
    beta = 0.001
    max_iters = 10000
    print('Gradient: Minimum found at ', find_minimum(X, beta, max_iters, gradient_search_direction)[0])
    print('Newton: Minimum found at ', find_minimum(X, beta, max_iters, newton_search_direction)[0])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# Generate fractal-like images using Rule 30. Rule and size are controlled via
# global variables. Uses NumPy arrays to speed up computation. Shows generated
# image using Pyplot.

import numpy as np
import matplotlib.pyplot as plt
from time import time

RULE = 30
GENERATIONS = 300
WIDTH = GENERATIONS*2

def main():
    start_time = time()

    rules = np.array([int(i) for i in np.binary_repr(RULE, 8)])
    population = np.zeros((GENERATIONS, WIDTH), dtype=int)

    # initial state
    population[0, WIDTH // 2] = 1

    for g in range(GENERATIONS-1):
        for i in range(1,WIDTH-1):
            neighbors = population[g, i-1:i+2]
            population[g+1, i] = rules[7-np.dot(neighbors, [4, 2, 1])]

    stop_time = time()

    print(f"finished in {round(stop_time-start_time, 2)}s")

    plt.imshow(population, cmap="gray_r")
    plt.show()

main()

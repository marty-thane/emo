# Solving the travelling businessman problem using modified hill climbing
# algorithm. Loads an external JSON file. Uses NumPy arrays to speed up
# computation.
# Note on algo: Best route in a generation becomes the exclusive parent of the
# next generation. Best parent is selected as the optimal solution.

import json
import numpy as np
import matplotlib.pyplot as plt

JSON_FILE = "cities.json"
GENERATIONS = 1000
POPULATION_SIZE = 50
MUTATION_RATE = 0.35

def load(file: str) -> dict:
    """
    Loads a JSON file provided as an argument and returns it as a dictionary.
    """
    with open(file, "r") as j: return json.load(j)

def preprocess(cities: dict) -> tuple[list, np.ndarray]:
    """
    Extracts names and coordinates from loaded JSON and returns them as a
    list/ndarray pair.
    """
    names = list(cities)
    coords = [list(c.values()) for c in cities.values()]
    return names, np.array(coords)

def dist_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Generates a distance matrix for given city coords and returns it.
    """
    n = len(coords)
    matrix = np.zeros((n,n), dtype=np.int16)
    for i in range(n-1):
        for j in range(i+1,n):
            matrix[i][j] = sum((coords[i]-coords[j])**2)**0.5 # pythagorean theorem
    return matrix + matrix.T

def fitness_of(route: np.ndarray, distances: np.ndarray) -> int:
    """
    Calculates score of a route given distance matrix.
    """
    return sum([distances[route[i]][route[i+1]] for i in range(len(route)-1)])

def mutate(route: np.ndarray, p: float) -> np.ndarray:
    """
    Swaps adjacent cities in a route with given probability. Returns the
    mutated route. Is able to swap first and last city too by using modulo.
    """
    mutated = np.copy(route)
    for i in range(len(route)):
        if np.random.uniform() <= p:
            mutated[[i, (i+1)%(i+1)]] = mutated[[(i+1)%(i+1), i]] # wrap around
    return mutated

def sort(items: list, order: list) -> list:
    """
    Returns provided list sorted by a given order.
    """
    return [items[order[i]] for i in range(len(items))]

def display(names: list, coords: np.ndarray) -> None:
    """
    Draws the final route using mathplotlib.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    plt.style.use("bmh")
    plt.plot(x, y, "-o")
    for i, name in enumerate(names):
        plt.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha="center")

    plt.title("Optimal route between cities")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.show()

def main():
    cities = load(JSON_FILE)
    names, coords = preprocess(cities)
    distances = dist_matrix(coords)

    parent = np.random.permutation(len(coords))
    best_route = parent
    best_score = fitness_of(best_route, distances)
    for _ in range(GENERATIONS):
        routes = np.array([
            mutate(parent, MUTATION_RATE)
            for _ in range(POPULATION_SIZE)
            ], dtype=np.int16)
        scores = np.array([fitness_of(route, distances) for route in routes], dtype=np.int16)
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_route = routes[best_idx]
            best_score = scores[best_idx]
            print(f"route: {sort(names, best_route)}, length: {best_score}")
        parent = routes[best_idx]

    ordered_names = sort(names, best_route)
    ordered_coords = np.array(sort(list(coords), best_route))
    display(ordered_names, ordered_coords)

main()


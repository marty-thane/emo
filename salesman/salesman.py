# WORK IN PROGRESS!!!
# Solving the travelling businessman problem using hill climbing. Loads an
# external JSON file.

# TODO: cleanup, keep the best best found solution.

import json
import numpy as np
import matplotlib.pyplot as plt

JSON_FILE = "cities.json"
GENERATIONS = 500
POPULATION_SIZE = 20
MUTATION_RATE = 0.23

def load(file: str) -> dict:
    """
    Loads a JSON file provided as an argument and returns it as a dictionary.
    """
    with open(file, "r") as j: return json.load(j)

def split(cities: dict) -> tuple[list, np.ndarray]:
    """
    Extracts names and coordinates from loaded JSON and returns them as a pair
    of lists.
    """
    names = list(cities)
    coords = np.array([list(c.values()) for c in cities.values()], dtype=np.int16)
    return names, coords

def dist(a: list, b: list) -> float:
    """
    Calculates distance between two points given as lists of coordinates using
    pythagorean theorem.
    """
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def distance_of(cities: np.ndarray) -> np.ndarray:
    """
    Generates a distance matrix for given cities and returns it.
    """
    n = len(cities)

    matrix = np.zeros((n,n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1,n):
            matrix[i][j] = dist(cities[i], cities[j])
    return matrix + matrix.T

def generate_initial(n: int, k: int) -> np.ndarray:
    """
    Produces population of size n with its members of size k.
    """
    default = np.arange(k, dtype=np.int16)
    return np.array([np.random.permutation(default) for _ in range(n)], dtype=np.int16)


def fitness_of(route: np.ndarray, distances: np.ndarray) -> int:
    """
    Calculates score of a route given distance matrix.
    """
    return sum([distances[route[i]][route[i+1]] for i in range(len(route)-1)])

def mutate(route: np.ndarray, p_mut: float) -> np.ndarray:
    mutated = np.copy(route)
    for i in range(len(route)-1):
        if np.random.uniform() <= p_mut:
            mutated[[i, i+1]] = mutated[[i+1, i]]
    return mutated

def display(names: list, coords: np.ndarray) -> None:
    """
    Draws the final route.
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    print(x_coords)

    plt.style.use("bmh")
    plt.scatter(x_coords, y_coords, zorder=2)
    plt.plot(x_coords, y_coords, zorder=1)
    for i, name in enumerate(names):
        plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,10), ha="center")

    plt.title("Optimal route between cities")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.show()

def main():
    cities = load(JSON_FILE)
    names, coords = split(cities)
    distances = distance_of(coords)
    population = generate_initial(POPULATION_SIZE, len(names))
    best_route = None
    for _ in range(GENERATIONS):
        scores = [fitness_of(member, distances) for member in population]
        best_idx = np.argmin(scores)
        print(scores[best_idx])
        best_route = population[best_idx]
        population = np.array([mutate(best_route, MUTATION_RATE) for _ in range(POPULATION_SIZE)], dtype=np.int16)
    ordered_names = [names[best_route[i]] for i in range(len(names))] # maybe use zip instead
    ordered_coords = [coords[best_route[i]] for i in range(len(names))]
    display(ordered_names, np.array(ordered_coords))


main()


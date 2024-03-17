# WORK IN PROGRESS!!!
# Solving the travelling businessman problem using hill climbing. Loads an external JSON file.

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

JSON_FILE = "items.json"
GENERATIONS = 1000

def display(route, cities) -> None:
    """
    Draws the final route. THIS IS ONLY A DRAFT AND WILL CHANGE WHEN THE ARCHITECTURE IS DECIDED.
    """
    x_coords = [cities[city]["x"] for city in route]
    y_coords = [cities[city]["y"] for city in route]

    plt.scatter(x_coords, y_coords, color="red", zorder=2)
    plt.plot(x_coords, y_coords,color="blue", zorder=1)
    for i, name in enumerate(route):
        plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,10), ha="center")

    plt.title("Optimal route between cities")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.show()

def main():
    ...

main()

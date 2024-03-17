# Solving the knapsack problem using blind algorithm. Loads an external JSON
# file. Uses built-in python types like dict and list, which may be slow for
# large numbers of items.

import json
from random import randint

JSON_FILE = "items.json"
GENERATIONS = 1000
MAX_WEIGHT = 10

def load(file: str) -> dict:
    """
    Loads a JSON file provided as an argument and returns it as a dictionary.
    """
    with open(file, "r") as j: return json.load(j)

def generate_from(items: dict) -> list:
    """
    Returns a random genotype based on items provided as an argument.
    """
    return [randint(0, 1) for _ in range(len(items))]

def gamma(genotype: list, items: dict) -> dict:
    """
    Filters the items by a given genotype. Returns the filtered dict.
    """
    filtered_keys = [i for i, g in zip(items, genotype) if g]
    filtered_items = {key: items[key] for key in filtered_keys}
    return filtered_items

def fitness_of(fenotype: dict) -> int:
    """
    Calculates fitness of a particular fenotype, which it returns.
    """
    total_weight = sum([item["weight"] for item in fenotype.values()])
    if total_weight > MAX_WEIGHT:
        return 0

    total_value = sum([item["value"] for item in fenotype.values()])
    return total_value

def main():
    best_fenotype = {}
    best_fitness = 0

    items = load(JSON_FILE)
    for _ in range(GENERATIONS):
        genotype = generate_from(items)
        fenotype = gamma(genotype, items)
        fitness = fitness_of(fenotype)
        if fitness > best_fitness:
            best_fenotype = fenotype
            best_fitness = fitness
    print(f"items: {list(best_fenotype)}, value: {best_fitness}")

main()

# Solving the knapsack problem using genetic algorithm. Loads an external JSON
# file. Uses NumPy arrays to speed up computation. Uses custom class to store
# item information. Prints useful information during run time.

import json
import numpy as np
from time import time

class Item:
    def __init__(self, name: str, value: int, weight: float):
        self._name = name
        self._value = value
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return self._value

    @property
    def weight(self) -> float:
        return self._weight

def parse_json(file: str) -> dict:
    """
    Loads a JSON file provided as an argument and returns it as a dictionary.
    """
    with open(file, "r") as f: return json.load(f)

def itemize(dictionary: dict) -> list:
    """
    Converts loaded JSON dict into a list of Item objects, which it returns.
    """
    return [
            Item(
                name=key,
                value=values["value"],
                weight=values["weight"]
                )
            for key, values in dictionary.items()
            ]

def gamma(genotype: np.ndarray) -> list:
    """
    Filters items by a given genotype. Returns the filtered fenotype.
    """
    return [i for g, i in zip(genotype, ITEMS) if g]

def fitness(phenotype: list) -> int:
    """
    Evaluates given phenotype and returns its score.
    """
    total_weight = sum([item.weight for item in phenotype])
    if total_weight > MAX_WEIGHT:
        return 1
    total_value = sum([item.value for item in phenotype])
    return total_value

def select(genotypes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Uses roulette selection to pick a member from population
    """
    random_number = np.random.rand()
    cumulative_score = 0
    for idx, score in enumerate(scores):
        cumulative_score += score
        if cumulative_score >= random_number:
            return genotypes[idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """
    Returns children created using one-point crossover technique.
    """
    splitpoint = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:splitpoint], parent2[splitpoint:]))
    child2 = np.concatenate((parent2[:splitpoint], parent1[splitpoint:]))
    return child1, child2

def mutate(genotype: np.ndarray) -> np.ndarray:
    """
    Mutates given genotype uniformly bit-by-bit.
    """
    mut = np.copy(genotype) # do not modify original
    for idx in range(len(genotype)):
        if np.random.rand() <= MUTATION_RATE:
            mut[idx] = 1 - mut[idx]
    return mut

JSON_FILE = "items.json"
MAX_WEIGHT = 10
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.05
ITEMS = itemize(parse_json(JSON_FILE))

def main():
    print(f"loaded {len(ITEMS)} items, weight limit {MAX_WEIGHT}")
    print(f"{2**len(ITEMS)} possible solutions, will try {POPULATION_SIZE*GENERATIONS}")

    start_time = time()

    genotypes = np.random.randint(0, 2, (POPULATION_SIZE, len(ITEMS)))

    best_phenotype = []
    best_score = 0
    for _ in range(GENERATIONS):
        phenotypes = [gamma(g) for g in genotypes]
        scores = np.array([fitness(f) for f in phenotypes])

        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_phenotype = phenotypes[best_idx]
            best_score = scores[best_idx]

        scores = scores / np.sum(scores) # normalize
        genotypes = np.array([select(genotypes, scores) for _ in range(POPULATION_SIZE)])
        genotypes = np.reshape([crossover(genotypes[idx], genotypes[idx+1]) for idx in range(0, POPULATION_SIZE, 2)], (POPULATION_SIZE, len(ITEMS)))
        genotypes = np.array([mutate(g) for g in genotypes])

    stop_time = time()

    print(f"found soution with score {best_score}:")
    print(np.array([item.name for item in best_phenotype]))
    print(f"finished in {round(stop_time-start_time, 2)}s")

main()

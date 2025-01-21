import json

import numpy as np
from qiskit.quantum_info import state_fidelity, hellinger_distance
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon


def fidelityCalc(density_matrix1, density_matrix2):
    fidelity = state_fidelity(density_matrix1, density_matrix2, validate=False)

    return fidelity


def traceDist(density_matrix1, density_matrix2):
    dist = np.abs(density_matrix1 - density_matrix2).trace() / 2

    return dist


def parse_distribution(oracle_output, mutant_output):
    oracle_output = str(oracle_output).replace("'", "\"")
    mutant_output = str(mutant_output).replace("'", "\"")
    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)
    return expected, observed

def align_distributions(expected, observed):
    # Ensure both distributions have the same keys
    all_keys = set(expected.keys()).union(set(observed.keys()))
    for key in all_keys:
        if key not in expected:
            expected[key] = 0
        if key not in observed:
            observed[key] = 0
    return expected, observed, all_keys

def normalize_distribution(distribution):
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()}

def compareChisquare(oracle_output, mutant_output):

    expected, observed = parse_distribution(oracle_output, mutant_output)
    #expected, observed, _ = align_distributions(expected, observed)

    if set(expected.keys()) != set(observed.keys()):
        return 0
    # Add epsilon to prevent zero frequencies
    #epsilon = 1e-10
    #expected = {k: v + epsilon for k, v in expected.items()}
    # Sorting the dictionary by keys
    expected = dict(sorted(expected.items()))
    observed = dict(sorted(observed.items()))

    result = chisquare(list(observed.values()), list(expected.values()))
    return result[1]  # p-value

def getHellinger(oracle_output, mutant_output):
    expected, observed = parse_distribution(oracle_output, mutant_output)
    distance = hellinger_distance(expected, observed)
    return distance


def jensenShannonDivergence(oracle_output, mutant_output):
    expected, observed = parse_distribution(oracle_output, mutant_output)
    expected, observed, all_keys = align_distributions(expected, observed)

    expected = normalize_distribution(expected)
    observed = normalize_distribution(observed)

    p_values = [expected[k] for k in all_keys]
    q_values = [observed[k] for k in all_keys]

    distance = jensenshannon(p_values, q_values)
    return distance

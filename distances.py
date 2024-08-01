import json
import ast
import re

import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix, state_fidelity, hellinger_distance, Statevector
from scipy.stats import chisquare

def compareChisquare(oracle_output, mutant_output):
    oracle_output = str(oracle_output)
    oracle_output = oracle_output.replace("'", "\"")
    mutant_output = str(mutant_output)
    mutant_output = mutant_output.replace("'", "\"")

    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)

    sorted_expected = dict(sorted(expected.items()))
    sorted_observed = dict(sorted(observed.items()))
    if len(list(sorted_observed.values())) == len(list(sorted_expected.values())):
        if sorted_expected.keys() == sorted_observed.keys():
            result = chisquare(list(sorted_observed.values()), list(sorted_expected.values()))
            result = result[1]
        else:
            result = 0
    else:
        result = 0

    return result


# Step 3: Parse the matrix strings to convert them into NumPy arrays
def parse_density_matrix(matrix_str):
    # Extract the actual matrix string
    start = matrix_str.find('[')
    end = matrix_str.rfind(']') + 1
    matrix_str = matrix_str[start:end]

    # Convert the string representation of the matrix to a list of lists
    matrix_list = ast.literal_eval(matrix_str)

    # Convert the list of lists to a NumPy array
    matrix_array = np.array(matrix_list)

    density_matrix = DensityMatrix(matrix_array)
    return density_matrix


def fidelityCalc(density_matrix1, density_matrix2):

    #density_matrix1 = parse_density_matrix(density_matrix_str1)
    #density_matrix2 = parse_density_matrix(density_matrix_str2)
    fidelity = state_fidelity(density_matrix1, density_matrix2, validate=False)

    return fidelity


def traceDist(density_matrix1, density_matrix2):
    #density_matrix1 = parse_density_matrix(density_matrix_str1)
    #density_matrix2 = parse_density_matrix(density_matrix_str2)

    dist = np.abs(density_matrix1-density_matrix2).trace()/2

    return dist


def getHellinger(oracle_output, mutant_output):
    oracle_output = str(oracle_output)
    oracle_output = oracle_output.replace("'", "\"")
    mutant_output = str(mutant_output)
    mutant_output = mutant_output.replace("'", "\"")

    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)

    distance = hellinger_distance(expected, observed)

    return distance



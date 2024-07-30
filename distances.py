import json
import ast
import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix, state_fidelity, hellinger_distance, Statevector
from scipy.stats import chisquare


# def compareDistResults(results_df):
#     # test_cases = 0
#     avg_result = 0
#     for inp in inputs:
#         oracle_ouput = oracle_df[oracle_df['input'] == str("'" + inp + "'")]
#         mutant_output = results_df[results_df['input'] == str("'" + inp + "'")]
#         # result = compareOutputs(oracle_ouput['counts'].values[0], mutant_output['counts'].values[0])
#         # if killed:
#         result = getHellinger(oracle_ouput['counts'].values[0], mutant_output['counts'].values[0])
#         avg_result = avg_result + result
#
#     avg_result = avg_result / len(inputs)
#
#     return avg_result
#
# def compareStateVectors(origin_stateVectors, mutant_stateVectors):
#     column_names = ['qc_name', 'input', 'output', 'fidelity', 'trace_dist']
#     results_df = pd.DataFrame(columns=column_names)
#     for ind, row in mutant_stateVectors.iterrows():
#         match = origin_stateVectors[origin_stateVectors['input'] == row.input]
#         if not match.empty:
#             index = match.index[0]
#             oracle_output = origin_stateVectors.at[index, 'statevector']
#             mutant_output = row.statevector
#             fidelity = fidelityCalc(oracle_output, mutant_output)
#             trace_dist = traceDist(oracle_output,mutant_output)
#             # tolerance = 1 - 1e-5
#             # if fidelity < tolerance:
#             #     killed = True
#             # else:
#             #     killed = False
#             new_line = {'qc_name': row.qc_name, 'input': row.input,'output':mutant_output, 'fidelity': fidelity, 'trace_dist': trace_dist}
#             new_df = pd.DataFrame.from_dict(new_line, orient='index').T
#             results_df = pd.concat([results_df, new_df], ignore_index=True)
#
#     return results_df

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


def fidelityCalc(stateVector1, stateVector2):
    stateVector1 = ast.literal_eval(stateVector1)
    stateVector2 = ast.literal_eval(stateVector2)
    stateVector1 = DensityMatrix(stateVector1)
    stateVector2 = DensityMatrix(stateVector2)

    fidelity = state_fidelity(stateVector1, stateVector2)

    return fidelity

def traceDist(stateVector1, stateVector2):
    stateVector1 = ast.literal_eval(stateVector1)
    stateVector2 = ast.literal_eval(stateVector2)
    stateVector1 = DensityMatrix(stateVector1)
    stateVector2 = DensityMatrix(stateVector2)

    dist = np.abs(stateVector1-stateVector2).trace()/2

    return dist

# def compareOutputs(oracle_output, mutant_output):
#     #result = False
#
#     oracle_output = str(oracle_output)
#     oracle_output = oracle_output.replace("'", "\"")
#     mutant_output = str(mutant_output)
#     mutant_output = mutant_output.replace("'", "\"")
#
#     expected = json.loads(oracle_output)
#     observed = json.loads(mutant_output)
#     # expected = ast.literal_eval(oracle_output)
#     # observed = ast.literal_eval(mutant_output)
#
#     sorted_expected = dict(sorted(expected.items()))
#     sorted_observed = dict(sorted(observed.items()))
#
#     if len(list(sorted_observed.values())) == len(list(sorted_expected.values())):
#         if sorted_expected.keys() == sorted_observed.keys():
#             results = chisquare(list(sorted_observed.values()), list(sorted_expected.values()))
#             result = results[1]
#             # if results[1] < 0.01:
#             #     result = True
#         else:
#             result = 0
#     else:
#         result = 0
#
#
#     return result

def getHellinger(oracle_output, mutant_output):
    oracle_output = str(oracle_output)
    oracle_output = oracle_output.replace("'", "\"")
    mutant_output = str(mutant_output)
    mutant_output = mutant_output.replace("'", "\"")

    expected = json.loads(oracle_output)
    observed = json.loads(mutant_output)

    distance = hellinger_distance(expected, observed)

    return distance



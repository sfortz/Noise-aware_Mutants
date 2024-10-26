import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files
from distances import fidelityCalc, traceDist, getHellinger, compareChisquare, jensenShannonDivergence


def get_theoretical_distribution(density_matrix, nb_shots):
    # Extract the diagonal elements (probabilities)
    probabilities = np.real(np.diag(density_matrix.data))
    probabilities = probabilities.copy()  # Create a writable copy

    # Normalize probabilities
    total_prob = np.sum(probabilities)
    if total_prob > 0:
        probabilities /= total_prob  # Normalize to ensure probabilities sum to 1

    # Calculate the number of qubits from the size of the density matrix
    num_qubits = int(np.log2(len(probabilities)))

    # Generate state labels for computational basis states
    state_labels = [format(i, f'0{num_qubits}b') for i in range(2 ** num_qubits)]

    # Calculate the initial counts for each state based on the number of shots
    initial_counts = np.array([prob * nb_shots for prob in probabilities])  # Use NumPy for easy operations

    # Round counts to nearest int and convert to Python int
    counts = [int(round(count)) for count in initial_counts]

    # Ensure the total counts sum to nb_shots
    total_counts = sum(counts)

    if total_counts != nb_shots:
        # Calculate the difference
        difference = nb_shots - total_counts

        # Calculate the adjustment needed
        adjustment_indices = np.argsort(initial_counts - counts)[:abs(difference)]

        # Adjust counts
        if difference > 0:
            for idx in adjustment_indices:
                counts[idx] += 1  # Increment counts for excess shots
        else:
            for idx in adjustment_indices:
                counts[idx] -= 1  # Decrement counts for excess counts

    # Create a dictionary mapping states to their corresponding counts
    count_dict = {state: int(count) for state, count in zip(state_labels, counts)}

    return count_dict


def get_ideal_thresholds(oracle_data):
    column_names = ['Name', 'Input', 'Chisquare', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation']
    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    results = []
    for key, value in oracle_lookup.items():

        # Theoretical distribution
        theoretical_distribution = get_theoretical_distribution(value['Ideal_density_matrix'], 10000)

        # Observed distribution (from Ideal_output_distribution)
        observed_distribution = value['Ideal_output_distribution']

        hellinger = getHellinger(theoretical_distribution, observed_distribution)
        jensen = jensenShannonDivergence(theoretical_distribution, observed_distribution)
        chisquare = compareChisquare(theoretical_distribution, observed_distribution)

        #fidelity = fidelityCalc(value['Theoretical_output_distribution'], value['Ideal_density_matrix'])
        #trace = traceDist(value['Theoretical_output_distribution'], value['Ideal_density_matrix'])
        #expectation = abs(value['Theoretical_output_distribution'] - value['Ideal_expectation_value'])
        name = value['Name'].split('/')[-1]
        fidelity = 0
        trace = 0
        expectation = 0
        new_line = {'Name': name, 'Input': value['Input'], 'Chisquare': chisquare, 'Hellinger': hellinger,
                    'Jensenshannon': jensen, 'Trace': trace, 'Fidelity': fidelity, 'Expectation': expectation}
        results.append(new_line)

    # Convert the results list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results, columns=column_names)
    return results_df


def process_files(service, origin_id):
    origin_files = get_files(service, origin_id)
    df_total = pd.DataFrame(
        columns=['Name', 'Input', 'Chisquare', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation'])
    for item in tqdm(origin_files, desc="Checking results..."):
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                pattern = r"indep_qiskit_|_output|.pkl"
                circuit_name = re.sub(pattern, "", filename)
                oracle_pkl = load_pickle_content(service, file_id)
                if isinstance(oracle_pkl, list):
                    new_df = get_ideal_thresholds(oracle_pkl)
                    df_total = pd.concat([df_total, new_df], ignore_index=True)
                else:
                    print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')
    print(df_total)
    mean_values = df_total.iloc[:, 2:].mean()
    print('Mean: ')
    print(mean_values)
    std_dev = df_total.iloc[:, 2:].std()
    print('Standard deviation: ')
    print(std_dev)
    n = len(df_total)  # Number of observations
    std_error = std_dev / np.sqrt(n)
    print('Standard error: ')
    print(std_error)
    print('Threshold: ')
    print(f"Chisquare: {mean_values['Chisquare'] + std_error['Chisquare']}")
    print(f"Hellinger: {mean_values['Hellinger'] + std_error['Hellinger']}")
    print(f"Jensenshannon: {mean_values['Jensenshannon'] + std_error['Jensenshannon']}")
    print(f"Trace: {mean_values['Trace'] + std_error['Trace']}")
    print(f"Fidelity: {(1 - mean_values['Fidelity']) + std_error['Fidelity']}")
    print(f"Expectation: {mean_values['Expectation'] + std_error['Expectation']}")


def main():
    origin_id = "1MTTleRgnFJ2UnYmbpzZoh2ndmWBJ3YJk"
    service = authenticate_google_drive()
    process_files(service, origin_id)


if __name__ == "__main__":
    main()

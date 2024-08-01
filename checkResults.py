import sys
import os
import re
import pickle
import pandas as pd
from tqdm import tqdm
from distances import fidelityCalc, traceDist, getHellinger, compareChisquare


def load_and_merge_files(folder):
    merged_data = []
    # Iterate through the folder
    for filename in os.listdir(folder):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                merged_data.extend(data)

    return merged_data


def calculate_killed_flags(ideal, noisy, tolerance_values):
    """
    Determines the killed flags based on ideal and noisy values and tolerance values.
    """
    killed_flags = {}
    killed_flags['Killed_IF'] = ideal['fidelity'] < tolerance_values['fidelity']
    killed_flags['Killed_NF'] = noisy['fidelity'] < tolerance_values['fidelity']
    killed_flags['Killed_IT'] = ideal['trace'] > tolerance_values['trace']
    killed_flags['Killed_NT'] = noisy['trace'] > tolerance_values['trace']
    killed_flags['Killed_IH'] = ideal['hellinger'] > tolerance_values['hellinger']
    killed_flags['Killed_NH'] = noisy['hellinger'] > tolerance_values['hellinger']
    killed_flags['Killed_IC'] = ideal['chisquare'] < tolerance_values['chisquare']
    killed_flags['Killed_NC'] = noisy['chisquare'] < tolerance_values['chisquare']
    return killed_flags


def checkResults(oracle_data, mutants_data):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity', 'Noisy_fidelity', 'Killed_IC', 'Killed_NC',
                    'Killed_IH', 'Killed_NH', 'Killed_IT', 'Killed_NT', 'Killed_IF', 'Killed_NF']

    results = []

    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    # Define tolerance values
    tolerance_values = {
        'fidelity': 1 - 1e-5,
        'trace': 1e-5,
        'hellinger': 0.05,
        'chisquare': 0.01
    }

    for mutant in mutants_data:
        input_value = mutant['Input']
        if input_value in oracle_lookup:
            oracle_entry = oracle_lookup[input_value]

            # Perform calculations
            ideal_chisquare = compareChisquare(oracle_entry['Ideal_output_distribution'],
                                               mutant['Ideal_output_distribution'])
            noisy_chisquare = compareChisquare(oracle_entry['Noisy_output_distribution'],
                                               mutant['Ideal_output_distribution'])

            ideal_hellinger = getHellinger(oracle_entry['Ideal_output_distribution'],
                                           mutant['Ideal_output_distribution'])
            noisy_hellinger = getHellinger(oracle_entry['Noisy_output_distribution'],
                                           mutant['Ideal_output_distribution'])

            ideal_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_fidelity = fidelityCalc(oracle_entry['Noisy_density_matrix'], mutant['Noisy_density_matrix'])

            ideal_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_trace = traceDist(oracle_entry['Noisy_density_matrix'], mutant['Noisy_density_matrix'])

            # Determine killed flags
            killed_flags = calculate_killed_flags(
                ideal={'fidelity': ideal_fidelity, 'trace': ideal_trace, 'hellinger': ideal_hellinger,
                       'chisquare': ideal_chisquare},
                noisy={'fidelity': noisy_fidelity, 'trace': noisy_trace, 'hellinger': noisy_hellinger,
                       'chisquare': noisy_chisquare},
                tolerance_values=tolerance_values
            )

            # Create a dictionary for the result row
            new_line = {
                'Name': mutant['Name'].split('/')[-1],
                'Input': mutant['Input'],
                'Ideal_chisquare': ideal_chisquare,
                'Noisy_chisquare': noisy_chisquare,
                'Ideal_hellinger': ideal_hellinger,
                'Noisy_hellinger': noisy_hellinger,
                'Ideal_trace': ideal_trace,
                'Noisy_trace': noisy_trace,
                'Ideal_fidelity': ideal_fidelity,
                'Noisy_fidelity': noisy_fidelity,
                **killed_flags
            }

            results.append(new_line)

    # Convert the results list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results, columns=column_names)

    return results_df


def main():
    origin_path = 'exec/origin_qc'
    all_mutants = 'exec/selected_mutants'

    #,Name,Input,Ideal_chisquare,Noisy_chisquare,Ideal_hellinger,Noisy_hellinger,Ideal_trace,Noisy_trace,Ideal_fidelity,Noisy_fidelity,Killed_IC,Killed_NC,Killed_IH,Killed_NH,Killed_IT,Killed_NT,Killed_IF,Killed_NF

    # Iterate through the folder
    for filename in tqdm(os.listdir(origin_path), desc="Checking results..."):
        if filename.endswith('.pkl'):
            file_path = os.path.join(origin_path, filename)
            # Pattern to match any of the substrings
            pattern = r"indep_qiskit_|_output|.pkl"
            # Remove the substrings
            circuit_name = re.sub(pattern, "", filename)
            with open(file_path, 'rb') as file:
                oracle_pkl = pickle.load(file)

            if isinstance(oracle_pkl, list):
                mutants_path = f'{all_mutants}/mutants_{circuit_name}'
                mutants_pkl = load_and_merge_files(mutants_path)
                results_df = checkResults(oracle_pkl, mutants_pkl)
                results_df.to_csv(f'results/results_{circuit_name}.csv')
            else:
                print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                sys.exit(1)


if __name__ == "__main__":
    main()

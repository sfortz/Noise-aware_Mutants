import os
import pickle
import re
import sys

import pandas as pd
from tqdm import tqdm
import itertools

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files
from distances import fidelityCalc, traceDist, getHellinger, compareChisquare, jensenShannonDivergence


def calculate_killed_flags(ideal, noisy, tolerance_values_ideal, tolerance_values_noisy):
    """
    Determines the killed flags based on ideal and noisy values and tolerance values.
    """
    killed_flags = {}
    killed_flags['Killed_IF'] = ideal['fidelity'] < tolerance_values_ideal['fidelity']
    killed_flags['Killed_NF'] = noisy['fidelity'] < tolerance_values_noisy['fidelity']
    killed_flags['Killed_IT'] = ideal['trace'] > tolerance_values_ideal['trace']
    killed_flags['Killed_NT'] = noisy['trace'] > tolerance_values_noisy['trace']
    killed_flags['Killed_IH'] = ideal['hellinger'] > tolerance_values_ideal['hellinger']
    killed_flags['Killed_NH'] = noisy['hellinger'] > tolerance_values_noisy['hellinger']
    killed_flags['Killed_IC'] = ideal['chisquare'] < tolerance_values_ideal['chisquare']
    killed_flags['Killed_NC'] = noisy['chisquare'] < tolerance_values_noisy['chisquare']
    killed_flags['Killed_IJ'] = ideal['jensenshannon'] < tolerance_values_ideal['jensenshannon']
    killed_flags['Killed_NJ'] = noisy['jensenshannon'] < tolerance_values_noisy['jensenshannon']
    return killed_flags


def check_results(oracle_data, mutants_data, tolerance_values_ideal, tolerance_values_noisy):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_jensenshannon', 'Noisy_jensenshannon', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity',
                    'Noisy_fidelity', 'Killed_IC', 'Killed_NC', 'Killed_IH', 'Killed_NH', 'Killed_IJ', 'Killed_NJ',
                    'Killed_IT', 'Killed_NT', 'Killed_IF', 'Killed_NF']

    results = []

    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

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

            ideal_jensenshannon = jensenShannonDivergence(oracle_entry['Ideal_output_distribution'],
                                               mutant['Ideal_output_distribution'])
            noisy_jensenshannon = jensenShannonDivergence(oracle_entry['Noisy_output_distribution'],
                                               mutant['Ideal_output_distribution'])

            ideal_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_fidelity = fidelityCalc(oracle_entry['Noisy_density_matrix'], mutant['Noisy_density_matrix'])

            ideal_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_trace = traceDist(oracle_entry['Noisy_density_matrix'], mutant['Noisy_density_matrix'])

            # Determine killed flags
            killed_flags = calculate_killed_flags(
                ideal={'fidelity': ideal_fidelity, 'trace': ideal_trace, 'hellinger': ideal_hellinger,
                       'chisquare': ideal_chisquare, 'jensenshannon': ideal_jensenshannon},
                noisy={'fidelity': noisy_fidelity, 'trace': noisy_trace, 'hellinger': noisy_hellinger,
                       'chisquare': noisy_chisquare, 'jensenshannon': noisy_jensenshannon},
                tolerance_values_ideal=tolerance_values_ideal,
                tolerance_values_noisy=tolerance_values_noisy
            )

            # Create a dictionary for the result row
            new_line = {
                'Name': mutant['Name'].split('/')[-1],
                'Input': mutant['Input'],
                'Ideal_chisquare': ideal_chisquare,
                'Noisy_chisquare': noisy_chisquare,
                'Ideal_hellinger': ideal_hellinger,
                'Noisy_hellinger': noisy_hellinger,
                'Ideal_jensenshannon': ideal_jensenshannon,
                'Noisy_jensenshannon': noisy_jensenshannon,
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


def get_files_id_dict(service, folder_id):
    items = get_files(service, folder_id)
    return {item['name']: item['id'] for item in items} if items else {}


def load_and_merge_files(service, folder_id):
    items = get_files(service, folder_id)
    merged_data = []

    for item in items:
        filename = item['name']
        file_id = item['id']

        try:
            data = load_pickle_content(service, file_id)
            merged_data.extend(data)
        except pickle.UnpicklingError:
            print(f'Error unpickling file: {filename}')
        except Exception as e:
            print(f'Error processing file {filename}: {str(e)}')

    return merged_data


def process_files(service, origin_id, mutants_id):
    # Define tolerance values
    tolerance_values_ideal = {
        'fidelity': 1 - 1e-5,
        'trace': 1e-5,
        'hellinger': 0.01,
        'jensenshannon': 0.01,
        'chisquare': 0.01
    }
    filelity_values = [1 - 1e-5, 1 - 5e-5]
    trace_values = [1e-5, 5e-5]
    hellinger_values = [0.01, 0.05]
    jensenshannon_values = [0.01, 0.05]
    chisquare_values = [0.01, 0.05]

    origin_files = get_files(service, origin_id)
    dic_mutant_folders = get_files_id_dict(service, mutants_id)

    for item in tqdm(origin_files, desc="Checking results..."):
        # Use itertools.product to get all possible combinations
        all_combinations = itertools.product(filelity_values, trace_values, hellinger_values, jensenshannon_values,
                                         chisquare_values)
        filename = item['name']
        file_id = item['id']
        print(filename)
        if filename.endswith('.pkl'):
            try:
                oracle_pkl = load_pickle_content(service, file_id)
                pattern = r"indep_qiskit_|_output|.pkl"
                circuit_name = re.sub(pattern, "", filename)
                if isinstance(oracle_pkl, list):
                    mutant_folder_id = dic_mutant_folders.get(f'mutants_{circuit_name}')
                    if mutant_folder_id:
                        #mutants_pkl = load_and_merge_files(service, mutant_folder_id)
                        for values in all_combinations:
                            fidelity_value, trace_value, hellinger_value, jensenshannon_value, chisquare_value = values
                            # Define tolerance values
                            tolerance_values_noisy = {
                                'fidelity': fidelity_value,
                                'trace': trace_value,
                                'hellinger': hellinger_value,
                                'jensenshannon': jensenshannon_value,
                                'chisquare': chisquare_value
                            }
                            print(tolerance_values_noisy)
                            #results_df = check_results(oracle_pkl, mutants_pkl, tolerance_values_ideal, tolerance_values_noisy)
                            #os.makedirs(f'results/results_{values}', exist_ok=True)
                            #results_df.to_csv(f'results/results_{values}/results_{circuit_name}.csv')
                    else:
                        print(f"No mutant folder found for {circuit_name}")
                else:
                    print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                    sys.exit(1)
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')


def main():
    origin_id = "1ScWuKuymtwcWabq_JG4OwLC18-2GUr3D"
    all_mutants_id = "1JUgQmxD0B7nFRxN3RagUgwwDH4GNrMqB"

    service = authenticate_google_drive()
    process_files(service, origin_id, all_mutants_id)


if __name__ == "__main__":
    main()

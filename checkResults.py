import os
import pickle
import re
import sys

import pandas as pd
from tqdm import tqdm

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
    killed_flags['Killed_IJ'] = ideal['jensenshannon'] > tolerance_values_ideal['jensenshannon']
    killed_flags['Killed_NJ'] = noisy['jensenshannon'] > tolerance_values_noisy['jensenshannon']
    killed_flags['Killed_IE'] = ideal['expectation'] > tolerance_values_ideal['expectation']
    killed_flags['Killed_NE'] = noisy['expectation'] > tolerance_values_noisy['expectation']
    return killed_flags

def check_results(oracle_data, mutants_data, tolerance_values_ideal, tolerance_values_noisy):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_jensenshannon', 'Noisy_jensenshannon', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity',
                    'Noisy_fidelity', 'Ideal_expectation', 'Noisy_expectation', 'Killed_IC', 'Killed_NC', 'Killed_IH', 'Killed_NH', 'Killed_IJ', 'Killed_NJ',
                    'Killed_IT', 'Killed_NT', 'Killed_IF', 'Killed_NF', 'Killed_IE', 'Killed_NE']

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
            noisy_chisquare = compareChisquare(oracle_entry['Ideal_output_distribution'],
                                               mutant['Noisy_output_distribution'])

            ideal_hellinger = getHellinger(oracle_entry['Ideal_output_distribution'],
                                           mutant['Ideal_output_distribution'])
            noisy_hellinger = getHellinger(oracle_entry['Ideal_output_distribution'],
                                           mutant['Noisy_output_distribution'])

            ideal_jensenshannon = jensenShannonDivergence(oracle_entry['Ideal_output_distribution'],
                                               mutant['Ideal_output_distribution'])
            noisy_jensenshannon = jensenShannonDivergence(oracle_entry['Ideal_output_distribution'],
                                               mutant['Noisy_output_distribution'])

            ideal_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Noisy_density_matrix'])

            ideal_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Noisy_density_matrix'])

            ideal_expectation = abs(oracle_entry['Ideal_expectation_value']-mutant['Ideal_expectation_value'])
            noisy_expectation = abs(oracle_entry['Ideal_expectation_value']-mutant['Noisy_expectation_value'])

            # ideal_chisquare = 0
            # noisy_chisquare = 0
            # ideal_hellinger = 0
            # noisy_hellinger = 0
            # ideal_jensenshannon = 0
            # noisy_jensenshannon = 0
            # ideal_fidelity = 0
            # noisy_fidelity = 0
            # ideal_trace = 0
            # noisy_trace = 0
            # ideal_expectation = 0
            # noisy_expectation = 0

            # Determine killed flags
            killed_flags = calculate_killed_flags(
                ideal={'fidelity': ideal_fidelity, 'trace': ideal_trace, 'hellinger': ideal_hellinger,
                       'chisquare': ideal_chisquare, 'jensenshannon': ideal_jensenshannon, 'expectation': ideal_expectation},
                noisy={'fidelity': noisy_fidelity, 'trace': noisy_trace, 'hellinger': noisy_hellinger,
                       'chisquare': noisy_chisquare, 'jensenshannon': noisy_jensenshannon, 'expectation': noisy_expectation},
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
                'Ideal_expectation': ideal_expectation,
                'Noisy_expectation': noisy_expectation,
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


def getModelTolerance(model):
    if model == 'brisbane':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9818071588272935,
            'trace': 0.9474790361650993,
            'hellinger': 0.9001136659697,
            'jensenshannon': 0.7718372511093367,
            'chisquare': 8.822528185214266e-158,
            'expectation': 0.7070786758337857
        }
    elif model == 'sheerbroke':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9817918801809562,
            'trace': 0.9165467778554671,
            'hellinger': 0.8723835633308122,
            'jensenshannon': 0.7525100350370049,
            'chisquare': 3.1954543753995917e-141,
            'expectation': 0.5896550492107876
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9817973573897236,
            'trace': 0.9134759188590066,
            'hellinger': 0.8760278316636461,
            'jensenshannon': 0.7549426398105366,
            'chisquare': 5.687357104528032e-162,
            'expectation': 0.6140893479852809
        }
    return tolerance_values_noisy


def process_files(service, origin_id, mutants_id, model, mutant):
    # Define tolerance values
    tolerance_values_ideal = {
        'fidelity': 1 - 1e-14,
        'trace': 1e-13,
        'hellinger': 0.13455009062719828,
        'jensenshannon': 0.11716009455796059,
        'chisquare': 0.318714816155845,
        'expectation': 0
    }
    possible_thresholds = ['A', 'I', 'N', 0.8, 0.5, 0.1]
    origin_files = get_files(service, origin_id)
    dic_mutant_folders = get_files_id_dict(service, mutants_id)

    for item in tqdm(origin_files, desc="Checking results..."):
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                pattern = r"indep_qiskit_|_output|.pkl"
                circuit_name = re.sub(pattern, "", filename)
                qubits = int(circuit_name.split('_')[1])
                if qubits <= 8:
                    print(circuit_name)
                    oracle_pkl = load_pickle_content(service, file_id)
                    if isinstance(oracle_pkl, list):
                        if mutant =='equiv':
                            mutant_folder_id = dic_mutant_folders.get(f'selected_equivalent_mutants_{circuit_name}')
                        else:
                            mutant_folder_id = dic_mutant_folders.get(f'mutants_{circuit_name}')
                        if mutant_folder_id:
                            mutants_pkl = load_and_merge_files(service, mutant_folder_id)
                            for threshold in possible_thresholds:
                                # Define tolerance values
                                if threshold == 'I':
                                    tolerance_values_noisy = tolerance_values_ideal
                                elif threshold == 'N':
                                    tolerance_values_noisy = getModelTolerance(model)
                                elif threshold == 'A':
                                    tolerance_values_noisy = getModelTolerance(model)
                                    tolerance_values_noisy = {
                                        'fidelity': 1 - ((1 - tolerance_values_noisy['fidelity']) + (1 - tolerance_values_ideal['fidelity'])),
                                        'trace': tolerance_values_noisy['trace'] + tolerance_values_ideal['trace'],
                                        'hellinger': tolerance_values_noisy['hellinger'] + tolerance_values_ideal['hellinger'],
                                        'jensenshannon': tolerance_values_noisy['jensenshannon'] + tolerance_values_ideal['jensenshannon'],
                                        'chisquare': tolerance_values_noisy['chisquare'] + tolerance_values_ideal['chisquare'],
                                        'expectation': tolerance_values_noisy['expectation'] + tolerance_values_ideal['expectation']
                                    }
                                else:
                                    tolerance_values_noisy = {
                                        'fidelity': 1 - threshold,
                                        'trace': threshold,
                                        'hellinger': threshold,
                                        'jensenshannon': threshold,
                                        'chisquare': threshold,
                                        'expectation': threshold
                                    }
                                results_df = check_results(oracle_pkl, mutants_pkl, tolerance_values_ideal, tolerance_values_noisy)
                                os.makedirs(f'results_{model}/results_{mutant}_{threshold}', exist_ok=True)
                                results_df.to_csv(f'results_{model}/results_{mutant}_{threshold}/results_{circuit_name}.csv')
                        else:
                            print(f"No mutant folder found for {circuit_name}")
                    else:
                        print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                        sys.exit(1)
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')


# If you obtain a Google authentication error, just delete the tocken.pickle file.
def main():
    models = ['brisbane', 'sheerbroke', 'kyiv']
    mutants = ['equiv', 'normal']
    for model in models:
        for mutant in mutants:
            if model == 'brisbane':
                origin_id = "1MTTleRgnFJ2UnYmbpzZoh2ndmWBJ3YJk"
                if mutant == 'equiv':
                    all_mutants_id = "1TuXmlQAARKVeOm4nTWSJBmI500Tns4aZ"
                else:
                    all_mutants_id = "1DlLaLyxSD5c0C1MqkCDlbrFvNrNLPo3e"
            elif model == 'sheerbroke':
                origin_id = "1mw2IXGwDlNYBaJ257fTWvbgn6uIFR_GE"
                if mutant == 'equiv':
                    all_mutants_id = "1LD99TCdLYlvdueVw3lFDS095LQnXHDXZ"
                else:
                    all_mutants_id = "1O6AGfEN3jXT2wIJv8Cdm639jdBTzr2tC"
            elif model == 'kyiv':
                origin_id = "1ZIiXv5wI-YjaxKaGR4CvXvfAKft-UwWJ"
                if mutant == 'equiv':
                    all_mutants_id = "1OsML98uRWNy-TKAQvQs-7bDGjrTne649"
                else:
                    all_mutants_id = "1Tqrv71qeMNIYnrW0CItiifaRPZO3h6hc"
            service = authenticate_google_drive()
            process_files(service, origin_id, all_mutants_id, model, mutant)

if __name__ == "__main__":
    main()

import os
import pickle
import re
import sys

import pandas as pd
from tqdm import tqdm

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files
from distances import fidelityCalc, traceDist, getHellinger, compareChisquare, jensenShannonDivergence

# Calculate tolerance values for various models
MODEL_TOLERANCES = {
    'brisbane': {
        'fidelity': 1 - 0.9818071588272935,
        'trace': 0.9474790361650993,
        'hellinger': 0.9001136659697,
        'jensenshannon': 0.7718372511093367,
        'chisquare': 8.822528185214266e-158,
        'expectation': 0.7070786758337857
    },
    'sheerbroke': {
        'fidelity': 1 - 0.9817918801809562,
        'trace': 0.9165467778554671,
        'hellinger': 0.8723835633308122,
        'jensenshannon': 0.7525100350370049,
        'chisquare': 3.1954543753995917e-141,
        'expectation': 0.5896550492107876
    },
    'kyiv': {
        'fidelity': 1 - 0.9817973573897236,
        'trace': 0.9134759188590066,
        'hellinger': 0.8760278316636461,
        'jensenshannon': 0.7549426398105366,
        'chisquare': 5.687357104528032e-162,
        'expectation': 0.6140893479852809
    }
}

# Ideal tolerance values (constant across all models)
TOLERANCE_VALUES_IDEAL = {
    'fidelity': 1 - 1e-14,
    'trace': 1e-13,
    'hellinger': 0.13455009062719828,
    'jensenshannon': 0.11716009455796059,
    'chisquare': 0.318714816155845,
    'expectation': 0
}


def generate_tolerance(threshold, model):
    # Check if threshold is numeric before using it in calculations
    if isinstance(threshold, (int, float)):
        return {
            'fidelity': 1 - threshold,
            'trace': threshold,
            'hellinger': threshold,
            'jensenshannon': threshold,
            'chisquare': threshold,
            'expectation': threshold
        }

    # Generate tolerance values based on the threshold.
    tolerance_noisy = {
        'I': TOLERANCE_VALUES_IDEAL,
        'N': MODEL_TOLERANCES.get(model, TOLERANCE_VALUES_IDEAL),
        'A': {k: v + TOLERANCE_VALUES_IDEAL[k] for k, v in MODEL_TOLERANCES.get(model, TOLERANCE_VALUES_IDEAL).items()}
    }

    # Return tolerance_noisy for string thresholds
    return tolerance_noisy.get(threshold, TOLERANCE_VALUES_IDEAL)


def calculate_killed_flags(ideal, noisy, tolerance_values_ideal, tolerance_values_noisy):
    """
    Determines the killed flags based on ideal and noisy values and tolerance values.
    """
    killed_flags = {'Killed_IF': ideal['fidelity'] < tolerance_values_ideal['fidelity'],
                    'Killed_NF': noisy['fidelity'] < tolerance_values_noisy['fidelity'],
                    'Killed_IT': ideal['trace'] > tolerance_values_ideal['trace'],
                    'Killed_NT': noisy['trace'] > tolerance_values_noisy['trace'],
                    'Killed_IH': ideal['hellinger'] > tolerance_values_ideal['hellinger'],
                    'Killed_NH': noisy['hellinger'] > tolerance_values_noisy['hellinger'],
                    'Killed_IC': ideal['chisquare'] < tolerance_values_ideal['chisquare'],
                    'Killed_NC': noisy['chisquare'] < tolerance_values_noisy['chisquare'],
                    'Killed_IJ': ideal['jensenshannon'] > tolerance_values_ideal['jensenshannon'],
                    'Killed_NJ': noisy['jensenshannon'] > tolerance_values_noisy['jensenshannon'],
                    'Killed_IE': ideal['expectation'] > tolerance_values_ideal['expectation'],
                    'Killed_NE': noisy['expectation'] > tolerance_values_noisy['expectation']}
    return killed_flags

def compute_metrics(oracle, mutant):
    """Compute and return separate dictionaries for ideal and noisy metrics."""
    ideal_metrics = {
        'chisquare': compareChisquare(oracle['Ideal_output_distribution'], mutant['Ideal_output_distribution']),
        'hellinger': getHellinger(oracle['Ideal_output_distribution'], mutant['Ideal_output_distribution']),
        'jensenshannon': jensenShannonDivergence(oracle['Ideal_output_distribution'],
                                                 mutant['Ideal_output_distribution']),
        'fidelity': fidelityCalc(oracle['Ideal_density_matrix'], mutant['Ideal_density_matrix']),
        'trace': traceDist(oracle['Ideal_density_matrix'], mutant['Ideal_density_matrix']),
        'expectation': abs(oracle['Ideal_expectation_value'] - mutant['Ideal_expectation_value'])
    }
    noisy_metrics = {
        'chisquare': compareChisquare(oracle['Ideal_output_distribution'], mutant['Noisy_output_distribution']),
        'hellinger': getHellinger(oracle['Ideal_output_distribution'], mutant['Noisy_output_distribution']),
        'jensenshannon': jensenShannonDivergence(oracle['Ideal_output_distribution'],
                                                 mutant['Noisy_output_distribution']),
        'fidelity': fidelityCalc(oracle['Ideal_density_matrix'], mutant['Noisy_density_matrix']),
        'trace': traceDist(oracle['Ideal_density_matrix'], mutant['Noisy_density_matrix']),
        'expectation': abs(oracle['Ideal_expectation_value'] - mutant['Noisy_expectation_value'])
    }
    return ideal_metrics, noisy_metrics


def check_results(oracle_data, mutants_data, tolerance_values_ideal, tolerance_values_noisy):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_jensenshannon', 'Noisy_jensenshannon', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity',
                    'Noisy_fidelity', 'Ideal_expectation', 'Noisy_expectation', 'Killed_IC', 'Killed_NC', 'Killed_IH',
                    'Killed_NH', 'Killed_IJ', 'Killed_NJ',
                    'Killed_IT', 'Killed_NT', 'Killed_IF', 'Killed_NF', 'Killed_IE', 'Killed_NE']
    results = []

    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    for mutant_entry in mutants_data:
        input_value = mutant_entry['Input']
        if input_value in oracle_lookup:
            oracle_entry = oracle_lookup[input_value]

            # Calling compute_metrics with separate outputs
            ideal_metrics, noisy_metrics = compute_metrics(oracle_entry, mutant_entry)
            # Determine killed flags
            killed_flags = calculate_killed_flags(ideal_metrics, noisy_metrics, tolerance_values_ideal,
                                                  tolerance_values_noisy)

            # Create a dictionary for the result row
            new_line = {
                'Name': mutant_entry['Name'].split('/')[-1],
                'Input': input_value,
                'Ideal_chisquare': ideal_metrics['chisquare'],
                'Noisy_chisquare': noisy_metrics['chisquare'],
                'Ideal_hellinger': ideal_metrics['hellinger'],
                'Noisy_hellinger': noisy_metrics['hellinger'],
                'Ideal_jensenshannon': ideal_metrics['jensenshannon'],
                'Noisy_jensenshannon': noisy_metrics['jensenshannon'],
                'Ideal_trace': ideal_metrics['trace'],
                'Noisy_trace': noisy_metrics['trace'],
                'Ideal_fidelity': ideal_metrics['fidelity'],
                'Noisy_fidelity': noisy_metrics['fidelity'],
                'Ideal_expectation': ideal_metrics['expectation'],
                'Noisy_expectation': noisy_metrics['expectation'],
                **killed_flags
            }
            results.append(new_line)

    # Convert to DataFrame, using headers as column names
    return pd.DataFrame(results, columns=column_names)


def process_files(service, origin_id, mutants_id, model, mutant):
    thresholds = ['A', 'I', 'N', 0.8, 0.5, 0.1]
    origin_files = get_files(service, origin_id)
    mutant_files = get_files(service, mutants_id)
    dic_mutant_folders = {item['name']: item['id'] for item in mutant_files} if mutant_files else {}

    # Pre-compute tolerance values for all thresholds
    tolerance_map = {threshold: generate_tolerance(threshold, model) for threshold in thresholds}

    for item in origin_files:
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                # Extract circuit name and number of qubits
                circuit_name = re.sub(r"indep_qiskit_|_output|.pkl", "", filename)

                if circuit_name not in {'qpeexact_8', 'vqe_8', 'wstate_8'}:
                    continue  # Skip if name do not match specific criteria

                oracle_pkl = load_pickle_content(service, file_id)

                if not isinstance(oracle_pkl, list):
                    print(f"Expected a list in oracle pickle file, but got {type(oracle_pkl)}.")
                    sys.exit(1)

                # Retrieve the mutant folder for the circuit
                if mutant == 'equiv':
                    mutant_folder_id = dic_mutant_folders.get(f'selected_equivalent_mutants_{circuit_name}')
                else:
                    mutant_folder_id = dic_mutant_folders.get(f'mutants_{circuit_name}')

                if not mutant_folder_id:
                    print(f"No mutant folder found for {circuit_name}")
                    continue

                # Process each mutant file in the folder
                mutants_items = get_files(service, mutant_folder_id)

                # for mutant_item in mutants_items:
                with tqdm(mutants_items, desc=f"Processing mutants for {circuit_name}", leave=True) as pbar:
                    for mutant_item in pbar:
                        mutant_file_id = mutant_item['id']
                        mutant_filename = mutant_item['name']
                        try:
                            mutant_data = load_pickle_content(service, mutant_file_id)

                            # Iterate over thresholds and save results
                            for threshold, tolerance_noisy in tolerance_map.items():
                                results_df = check_results(oracle_pkl, mutant_data, TOLERANCE_VALUES_IDEAL,
                                                           tolerance_noisy)
                                output_folder = f'results_{model}/results_{mutant}_{threshold}'
                                os.makedirs(output_folder, exist_ok=True)

                                # Check if file already exists to determine whether to write headers
                                results_csv_path = f'{output_folder}/results_{circuit_name}.csv'
                                write_header = not os.path.exists(results_csv_path)

                                # Append results with headers only if the file does not exist
                                results_df.to_csv(results_csv_path, mode='a', header=write_header, index=False)

                        except pickle.UnpicklingError:
                            print(f"Error unpickling mutant file: {mutant_filename}")
                        except Exception as e:
                            print(f"Error processing mutant file {mutant_filename}: {str(e)}")

            except pickle.UnpicklingError:
                print(f"Error unpickling file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")


# If you obtain a Google authentication error, just delete the tocken.pickle file.
def main():
    models = ['brisbane']  #, 'sheerbroke', 'brisbane']
    mutants = ['equiv']  #, 'normal']
    for model in models:
        print("============================================================================================")
        print("Executing model", model)
        print("============================================================================================")
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

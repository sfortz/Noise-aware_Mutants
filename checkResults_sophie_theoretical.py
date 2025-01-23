import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from qiskit.quantum_info import Operator
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


def get_theoretical_expectation_value(density_matrix):
    # Determine the number of qubits
    dim = density_matrix.data.shape[0]
    num_qubits = int(np.log2(dim))

    if 2 ** num_qubits != dim:
        raise ValueError("Density matrix size must be a power of 2 (2^n x 2^n).")

    # Create the multi-qubit Z operator
    single_qubit_z = np.array([[1, 0], [0, -1]])  # Z operator for one qubit
    z_operator = single_qubit_z  # Start with the single-qubit Z
    for _ in range(1, num_qubits):  # Extend for multi-qubit systems
        z_operator = np.kron(z_operator, single_qubit_z)

    # Convert Z operator to Qiskit Operator
    z_operator = Operator(z_operator)

    # Compute the expectation value: Tr(rho * Z)
    expectation_value = np.trace(density_matrix.data @ z_operator.data).real

    return expectation_value


def check_results(oracle_data, mutants_data, tolerance_values_ideal, tolerance_values_noisy):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_jensenshannon', 'Noisy_jensenshannon', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity',
                    'Noisy_fidelity', 'Ideal_expectation', 'Noisy_expectation', 'Killed_IC', 'Killed_NC', 'Killed_IH',
                    'Killed_NH', 'Killed_IJ', 'Killed_NJ',
                    'Killed_IT', 'Killed_NT', 'Killed_IF', 'Killed_NF', 'Killed_IE', 'Killed_NE']

    results = []

    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    for mutant in mutants_data:
        input_value = mutant['Input']
        if input_value in oracle_lookup:
            oracle_entry = oracle_lookup[input_value]

            # Theoretical distribution
            theoretical_distribution = get_theoretical_distribution(oracle_entry['Ideal_density_matrix'], 10000)

            # Observed distribution (from Ideal_output_distribution)
            observed_ideal_distribution = mutant['Ideal_output_distribution']

            ideal_hellinger = getHellinger(theoretical_distribution, observed_ideal_distribution)
            ideal_jensenshannon = jensenShannonDivergence(theoretical_distribution, observed_ideal_distribution)
            ideal_chisquare = 0 #compareChisquare(theoretical_distribution, observed_ideal_distribution)

            # Observed distribution (from Noisy_output_distribution)
            observed_noisy_distribution = mutant['Noisy_output_distribution']

            noisy_hellinger = getHellinger(theoretical_distribution, observed_noisy_distribution)
            noisy_jensenshannon = jensenShannonDivergence(theoretical_distribution, observed_noisy_distribution)
            noisy_chisquare = 1 #compareChisquare(theoretical_distribution, observed_noisy_distribution)

            theoretical_expectation_value = get_theoretical_expectation_value(oracle_entry['Ideal_density_matrix'])
            ideal_expectation = abs(theoretical_expectation_value - mutant['Ideal_expectation_value'])
            noisy_expectation = abs(theoretical_expectation_value - mutant['Noisy_expectation_value'])

            #ideal_expectation = abs(oracle_entry['Ideal_expectation_value'] - mutant['Ideal_expectation_value'])
            #noisy_expectation = abs(oracle_entry['Ideal_expectation_value'] - mutant['Noisy_expectation_value'])

            # Perform calculations
            #ideal_chisquare = compareChisquare(oracle_entry['Ideal_output_distribution'], mutant['Ideal_output_distribution'])
            #noisy_chisquare = compareChisquare(oracle_entry['Ideal_output_distribution'], mutant['Noisy_output_distribution'])

            #ideal_hellinger = getHellinger(oracle_entry['Ideal_output_distribution'], mutant['Ideal_output_distribution'])
            #noisy_hellinger = getHellinger(oracle_entry['Ideal_output_distribution'], mutant['Noisy_output_distribution'])

            #ideal_jensenshannon = jensenShannonDivergence(oracle_entry['Ideal_output_distribution'], mutant['Ideal_output_distribution'])
            #noisy_jensenshannon = jensenShannonDivergence(oracle_entry['Ideal_output_distribution'], mutant['Noisy_output_distribution'])

            ideal_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_fidelity = fidelityCalc(oracle_entry['Ideal_density_matrix'], mutant['Noisy_density_matrix'])

            ideal_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Ideal_density_matrix'])
            noisy_trace = traceDist(oracle_entry['Ideal_density_matrix'], mutant['Noisy_density_matrix'])

            # Determine killed flags
            killed_flags = calculate_killed_flags(
                ideal={'fidelity': ideal_fidelity, 'trace': ideal_trace, 'hellinger': ideal_hellinger,
                       'chisquare': ideal_chisquare, 'jensenshannon': ideal_jensenshannon,
                       'expectation': ideal_expectation},
                noisy={'fidelity': noisy_fidelity, 'trace': noisy_trace, 'hellinger': noisy_hellinger,
                       'chisquare': noisy_chisquare, 'jensenshannon': noisy_jensenshannon,
                       'expectation': noisy_expectation},
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
            'fidelity': 1 - 0.9118523593799848,
            'trace': 0.034415871303828394,
            'hellinger': 0.2130665216337895,
            'jensenshannon': 0.19154709011416926,
            'chisquare': 0.0,
            'expectation': 0.019239830427815005
        }

    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.7606042378746769,
            'trace': 0.09087481763504617,
            'hellinger': 0.2835557485569176,
            'jensenshannon': 0.26311290831221235,
            'chisquare': 0.0,
            'expectation': 0.032351843338062784
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9024474384739689,
            'trace': 0.04045065953149715,
            'hellinger': 0.19741965571741749,
            'jensenshannon': 0.1836415632471504,
            'chisquare': 0.0,
            'expectation': 0.0239542540495735
        }

    else:
        tolerance_values_noisy = {}

    return tolerance_values_noisy


def cap_value(value):
    return min(1, max(0, value))


def get_tolerance_values_noisy(model, threshold):
    tolerance_values_ideal = {
        'fidelity': 1 - 1e-14,
        'trace': 1e-13,
        'hellinger': 0.04178952039843151,
        'jensenshannon': 0.04058294316372258,
        'chisquare': 0.0,
        'expectation': 0.011574442770798709
    }

    # Define tolerance values
    if threshold == 'I':
        tolerance_values_noisy = tolerance_values_ideal
    elif threshold == 'N':
        tolerance_values_noisy = getModelTolerance(model)
    elif threshold == 'A':
        tolerance_values_noisy = getModelTolerance(model)
        tolerance_values_noisy = {
            'fidelity': cap_value(
                1 - ((1 - tolerance_values_noisy['fidelity']) + (1 - tolerance_values_ideal['fidelity']))),
            'trace': cap_value(tolerance_values_noisy['trace'] + tolerance_values_ideal['trace']),
            'hellinger': cap_value(tolerance_values_noisy['hellinger'] + tolerance_values_ideal['hellinger']),
            'jensenshannon': cap_value(
                tolerance_values_noisy['jensenshannon'] + tolerance_values_ideal['jensenshannon']),
            'chisquare': cap_value(tolerance_values_noisy['chisquare'] + tolerance_values_ideal['chisquare']),
            'expectation': cap_value(tolerance_values_noisy['expectation'] + tolerance_values_ideal['expectation'])
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

    return tolerance_values_noisy


def process_files(service, origin_id, mutants_id, model, mutant):
    # Define tolerance values
    possible_thresholds = ['A', 'I', 'N']  #, 0.8, 0.5, 0.1]
    tolerance_values_ideal = get_tolerance_values_noisy(model, 'I')
    dic_noisy_tolerance = {}
    for threshold in possible_thresholds:
        dic_noisy_tolerance[threshold] = get_tolerance_values_noisy(model, threshold)

    origin_files = get_files(service, origin_id)
    dic_mutant_folders = get_files_id_dict(service, mutants_id)

    for item in origin_files:
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                pattern = r"indep_qiskit_|_output|.pkl"
                circuit_name = re.sub(pattern, "", filename)
                qubits = int(circuit_name.split('_')[1])
                if (circuit_name in ['qpeexact_3','vqe_3','wstate_2']): #(qubits == 6) & (circuit_name in ['qpeexact_6','vqe_6']):  # ae_8, qft_8, wstate_8, vqe_8, qpeexact_8, qftentangled_8
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
                        sys.exit(1)

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
                                for threshold in possible_thresholds:
                                    tolerance_values_noisy = dic_noisy_tolerance[threshold]
                                    results_df = check_results(oracle_pkl, mutant_data, tolerance_values_ideal,
                                                               tolerance_values_noisy)
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
    models = ['kyiv']  #['brisbane', 'sherbrooke', 'kyiv']
    mutants = ['normal'] #'equiv', 'normal']
    for model in models:
        for mutant in mutants:
            print("============================================================================================")
            print(f"Executing {mutant} mutants on {model} simulator")
            print("============================================================================================")
            if model == 'brisbane':
                origin_id = "1IsrwIF2IKSYWCCHDR5oXPdF33Ml0dmRS"  # Change for Run not part of threshold definition
                if mutant == 'equiv':
                    all_mutants_id = "1CsNEuLnHUwn-4fJ4fpby-NTY49W0f_vK"
                else:
                    all_mutants_id = "1IHTG_PbpF2ayFLCEoEbEgN6tDzapsRnw"
            elif model == 'sherbrooke':
                origin_id = "15xelXYFEassEnLt3IiHEL0o5-oyM0RQa"  # Change for Run not part of threshold definition
                if mutant == 'equiv':
                    all_mutants_id = "1sSBRuM5sSPZp9BZAGMZkmaZh0wWrsMsg"
                else:
                    all_mutants_id = "1XexEZHJyL5oGDhMCq7Tkf-yfKoGsLuK5"
            elif model == 'kyiv':
                origin_id = "125Pd27lcQwg0S64uHZFewIAFOvO9AbJ1"  # Change for Run not part of threshold definition
                if mutant == 'equiv':
                    all_mutants_id = "1oep-q4iVAwikGnpU6XTpptpN1iezPvhL"
                else:
                    all_mutants_id = "1nmXYpzDvXoJR5aZT-g_LmxrAuWe48HRJ"
            else:
                origin_id = None
                all_mutants_id = None
            service = authenticate_google_drive()
            process_files(service, origin_id, all_mutants_id, model, mutant)


if __name__ == "__main__":
    main()

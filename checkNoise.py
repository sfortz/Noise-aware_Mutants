import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from qiskit.quantum_info import Operator
from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files
from distances import fidelityCalc, traceDist, getHellinger, jensenShannonDivergence


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

    if total_counts != nb_shots:  # TODO: Check if Qiskit can transform probabilities to shots directly.
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


# Objective: Print distance between noisy and ideal, not using any mutant-oracle comparison
# Compute distance between theoretical (from oracle) and ideal (from oracle)
# Compute distance between theoretical (from oracle) and noisy (from oracle)

# Compute distance between theoretical (mutant no noise) and noisy (mutant with noise)
# Compute distance between theoretical (mutant no noise) and ideal (mutant with noise)
# Print noisy

def check_results(data):
    column_names = ['Name', 'Input', 'Ideal_hellinger', 'Noisy_hellinger',
                    'Ideal_jensenshannon', 'Noisy_jensenshannon', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity',
                    'Noisy_fidelity', 'Ideal_expectation', 'Noisy_expectation']

    results = []

    for value in data:
        # Theoretical distribution
        theoretical_distribution = get_theoretical_distribution(value['Ideal_density_matrix'], 10000)

        # Observed distribution (from Ideal_output_distribution)
        observed_ideal_distribution = value['Ideal_output_distribution']

        ideal_hellinger = getHellinger(theoretical_distribution, observed_ideal_distribution)
        ideal_jensenshannon = jensenShannonDivergence(theoretical_distribution, observed_ideal_distribution)

        # Observed distribution (from Noisy_output_distribution)
        observed_noisy_distribution = value['Noisy_output_distribution']

        noisy_hellinger = getHellinger(theoretical_distribution, observed_noisy_distribution)
        noisy_jensenshannon = jensenShannonDivergence(theoretical_distribution, observed_noisy_distribution)

        theoretical_expectation_value = get_theoretical_expectation_value(value['Ideal_density_matrix'])
        ideal_expectation = abs(theoretical_expectation_value - value['Ideal_expectation_value'])
        noisy_expectation = abs(theoretical_expectation_value - value['Noisy_expectation_value'])

        ideal_fidelity = fidelityCalc(value['Ideal_density_matrix'], value['Ideal_density_matrix'])
        noisy_fidelity = fidelityCalc(value['Ideal_density_matrix'], value['Noisy_density_matrix'])

        ideal_trace = traceDist(value['Ideal_density_matrix'], value['Ideal_density_matrix'])
        noisy_trace = traceDist(value['Ideal_density_matrix'], value['Noisy_density_matrix'])

        # Create a dictionary for the result row
        new_line = {
            'Name': value['Name'].split('/')[-1],
            'Input': value['Input'],
            'Ideal_hellinger': ideal_hellinger,
            'Noisy_hellinger': noisy_hellinger,
            'Ideal_jensenshannon': ideal_jensenshannon,
            'Noisy_jensenshannon': noisy_jensenshannon,
            'Ideal_trace': ideal_trace,
            'Noisy_trace': noisy_trace,
            'Ideal_fidelity': ideal_fidelity,
            'Noisy_fidelity': noisy_fidelity,
            'Ideal_expectation': ideal_expectation,
            'Noisy_expectation': noisy_expectation
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


def process_files(service, dic_data_folder, output_folder):
    with tqdm(dic_data_folder, desc=f"Processing...", leave=True) as pbar:
        for item in pbar:
            filename = item['name']
            file_id = item['id']
            if filename.endswith('.pkl'):
                try:
                    pattern = r"indep_qiskit_|_output|.pkl"
                    circuit_name = re.sub(pattern, "", filename)
                    qubits = int(circuit_name.split('_')[1])
                    #if circuit_name:# (in ['qpeexact_3','vqe_3','wstate_2']): #(qubits == 6) & (circuit_name in ['qpeexact_6','vqe_6']):  # ae_8, qft_8, wstate_8, vqe_8, qpeexact_8, qftentangled_8
                    if qubits <= 8:
                        data_pkl = load_pickle_content(service, file_id)
                        if not isinstance(data_pkl, list):
                            print(f"Expected a list in oracle pickle file, but got {type(data_pkl)}.")
                            sys.exit(1)

                        results_df = check_results(data_pkl)
                        os.makedirs(output_folder, exist_ok=True)

                        # Check if file already exists to determine whether to write headers
                        results_csv_path = f'{output_folder}/results_{circuit_name}.csv'
                        write_header = not os.path.exists(results_csv_path)

                        # Append results with headers only if the file does not exist
                        results_df.to_csv(results_csv_path, mode='a', header=write_header, index=False)

                except pickle.UnpicklingError:
                    print(f"Error unpickling file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")


# If you obtain a Google authentication error, just delete the tocken.pickle file.
def main():
    models = ['brisbane', 'sherbrooke', 'kyiv']
    mutants = ['equiv', 'normal']
    for model in models:

        print("============================================================================================")
        print(f"Executing originals on {model} simulator")
        print("============================================================================================")
        if model == 'brisbane':
            origin_name = 'brisbane_origin'
            origin_id = "1HCneX79jzbIFpMeg33SUw4eMSLuIX0EK"  # Extra run folder id
        elif model == 'sherbrooke':
            origin_name = 'sherbrooke_origin'
            origin_id = "1cEU1SR50KOIfo1jTNXL3PhTZzvBKNoLZ"  # Change for Run not part of threshold definition
        elif model == 'kyiv':
            origin_name = 'kyiv_origin'
            origin_id = "1IQ8uEpPg7AxmIoW_nKEfZhAOPhdI9Itd"  # Change for Run not part of threshold definition
        else:
            origin_name = None
            origin_id = None
        service = authenticate_google_drive()
        dic_data_folder = get_files(service, origin_id)
        output_folder = f'results_TEST/results_{origin_name}'
        process_files(service, dic_data_folder, output_folder)

        for mutant in mutants:
            print("============================================================================================")
            print(f"Executing {mutant} mutants on {model} simulator")
            print("============================================================================================")
            if model == 'brisbane':
                if mutant == 'equiv':
                    origin_name = 'brisbane_equiv'
                    origin_id = "1XNnFqHmF5Fv3QXNaaKfZsActRX2S3pz3"  # equiv_qc folder id
                else:
                    origin_name = 'brisbane_normal'
                    origin_id = "1Nz8d3u_cf3HvRxSJ_e6PgYgmZCnjx5Td"  # mutant_qc folder id
            elif model == 'sherbrooke':
                if mutant == 'equiv':
                    origin_name = 'sherbrooke_equiv'
                    origin_id = "1sFNMH2ky6zhMtN8xFtpwYJV7T92MSkh5"
                else:
                    origin_name = 'sherbrooke_normal'
                    origin_id = "10W0wuoWfVH0FOh2LOxFXQXAuv5PvtKoB"
            elif model == 'kyiv':
                if mutant == 'equiv':
                    origin_name = 'kyiv_equiv'
                    origin_id = "1DpZSMM0aj8gP7K0XQ_weGCVRw3KltHfi"
                else:
                    origin_name = 'kyiv_normal'
                    origin_id = "1tpX9nl0vlus-0A5_wiDPCm0ITWZIBvoe"
            else:
                origin_id = None
                origin_name = None
            service = authenticate_google_drive()

            for mutant_folder_name, mutant_folder_id in get_files_id_dict(service, origin_id).items():
                # Process each mutant file in the folder
                print(f"Processing mutant {mutant_folder_name}")
                dic_data_folder = get_files(service, mutant_folder_id)
                output_folder = f'results_TEST/results_{origin_name}/{mutant_folder_id}'
                process_files(service, dic_data_folder, output_folder)


if __name__ == "__main__":
    main()

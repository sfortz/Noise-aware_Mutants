import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from qiskit.quantum_info import Operator
from tqdm import tqdm

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files, get_folders
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


def get_ideal_thresholds(oracle_data):
    column_names = ['Name', 'Input', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation']
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

        theoretical_expectation_value = get_theoretical_expectation_value(value['Ideal_density_matrix'])
        expectation = abs(theoretical_expectation_value - value['Ideal_expectation_value'])

        fidelity = 0  # Computed in getIdealThresholds.ipynb
        trace = 0  # Computed in getIdealThresholds.ipynb

        name = value['Name'].split('/')[-1]
        new_line = {'Name': name, 'Input': value['Input'], 'Hellinger': hellinger,
                    'Jensenshannon': jensen, 'Trace': trace, 'Fidelity': fidelity, 'Expectation': expectation}
        results.append(new_line)

    # Convert the results list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results, columns=column_names)
    return results_df


def get_noisy_thresholds(oracle_data):
    column_names = ['Name', 'Input', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation']
    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    results = []
    for key, value in oracle_lookup.items():
        # Theoretical distribution
        theoretical_distribution = get_theoretical_distribution(value['Ideal_density_matrix'], 10000)

        # Observed distribution (from Noisy_output_distribution)
        observed_distribution = value['Noisy_output_distribution']

        hellinger = getHellinger(theoretical_distribution, observed_distribution)
        jensen = jensenShannonDivergence(theoretical_distribution, observed_distribution)

        theoretical_expectation_value = get_theoretical_expectation_value(value['Ideal_density_matrix'])
        expectation = abs(theoretical_expectation_value - value['Noisy_expectation_value'])

        fidelity = fidelityCalc(value['Ideal_density_matrix'], value['Noisy_density_matrix'])
        trace = traceDist(value['Ideal_density_matrix'], value['Noisy_density_matrix'])

        name = value['Name'].split('/')[-1]
        new_line = {'Name': name, 'Input': value['Input'], 'Hellinger': hellinger,
                    'Jensenshannon': jensen, 'Trace': trace, 'Fidelity': fidelity, 'Expectation': expectation}
        results.append(new_line)

    results_df = pd.DataFrame(results, columns=column_names)

    return results_df


def is_processed(filename, path):
    """Check if the file has already been processed by looking for its temp file."""
    temp_file_path = os.path.join(path, f"{filename}.tmp")
    return os.path.exists(temp_file_path)


def save_temp_file(filename, data, path):
    """Save the processed data to a temporary file."""
    temp_file_path = os.path.join(path, f"{filename}.tmp")
    with open(temp_file_path, 'wb') as f:
        pickle.dump(data, f)


def load_temp_file(filename, path):
    """Load the processed data from a temporary file."""
    temp_file_path = os.path.join(path, filename)
    with open(temp_file_path, 'rb') as f:
        return pickle.load(f)


def process_files(service, origin_id, isNoisy, temp_dir):
    origin_files = get_files(service, origin_id)

    for item in tqdm(origin_files, desc="Processing files..."):
        filename = item['name']
        file_id = item['id']

        # Skip already processed files
        if is_processed(filename, temp_dir):
            #print(f"Skipping already processed file: {filename}")
            continue

        if filename.endswith('.pkl'):
            pattern = r"indep_qiskit_|_output|.pkl"
            circuit_name = re.sub(pattern, "", filename)
            qubits = int(circuit_name.split('_')[1])
            if qubits <= 8:
                try:
                    oracle_pkl = load_pickle_content(service, file_id)

                    if isinstance(oracle_pkl, list):
                        if isNoisy:
                            new_df = get_noisy_thresholds(oracle_pkl)
                        else:
                            new_df = get_ideal_thresholds(oracle_pkl)

                        # Save the processed data to a temporary file
                        save_temp_file(filename, new_df, temp_dir)
                    else:
                        print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                        sys.exit(1)
                except pickle.UnpicklingError:
                    print(f"Error unpickling file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")


def merge_all_temp_files(path):
    """Merge all temporary files into a single DataFrame."""

    all_files = [f for f in os.listdir(path) if f.endswith('.tmp')]
    data_frames = []
    for temp_file in all_files:

        pattern = r"indep_qiskit_|_output|.pkl.tmp"
        circuit_name = re.sub(pattern, "", temp_file)
        qubits = int(circuit_name.split('_')[1])

        if qubits < 9:
            temp_file_path = os.path.join(path, temp_file)
            try:
                with open(temp_file_path, 'rb') as f:
                    data = pickle.load(f)
                data_frames.append(data)
            except Exception as e:
                print(f"Error loading temporary file {temp_file}: {str(e)}")
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


def process_runs(service, folder_id, isNoisy, temp_dir):
    runs = get_folders(service, folder_id)
    runs_df_list = []
    for run in runs:
        run_id = run['id']
        temp_path = os.path.join(temp_dir, run_id)
        os.makedirs(temp_path, exist_ok=True)
        process_files(service, run_id, isNoisy, temp_path)
        # Merge all temporary files into a single DataFrame
        if os.path.isdir(temp_path):
            df_run = merge_all_temp_files(temp_path)
            runs_df_list.append(df_run)
    return runs_df_list


def display(runs_df_list):
    # Select numeric and non-numeric columns separately
    numeric_columns = runs_df_list[0].select_dtypes(include=[np.number]).columns
    non_numeric_columns = runs_df_list[0].select_dtypes(exclude=[np.number]).columns

    # Stack only the numeric columns of all DataFrames
    stacked_array = np.stack([df[numeric_columns].values for df in runs_df_list], axis=0)

    # Calculate the mean and standard deviation for the numeric columns
    mean_df = pd.DataFrame(np.mean(stacked_array, axis=0), columns=numeric_columns)
    std_df = pd.DataFrame(np.std(stacked_array, axis=0), columns=numeric_columns)

    # For non-numeric columns, just take the first DataFrame (since they are the same across all)
    non_numeric_df = runs_df_list[0][non_numeric_columns]

    # Combine the results: concatenate mean, std, and non-numeric columns
    # First, combine mean and std for numeric columns
    combined_numeric_df = pd.concat([mean_df, std_df], axis=1, keys=["mean", "std"])
    combined_numeric_df.columns = ['_'.join(map(str, col)) for col in combined_numeric_df.columns]

    # Now add non-numeric columns to the final result
    final_df = pd.concat([combined_numeric_df, non_numeric_df], axis=1)

    # print(final_df)
    values_3rd_quartile = final_df.iloc[:, :-2].quantile(0.75)
    values_1st_quartile = final_df.iloc[:, :-2].quantile(0.25)

    n = len(final_df)  # Number of observations
    print('----------------------------------------')
    print('Mean: ')
    print(values_3rd_quartile[['mean_Hellinger', 'mean_Jensenshannon', 'mean_Trace', 'mean_Fidelity',
                  'mean_Expectation']])
    print('----------------------------------------')
    print('Standard deviation: ')
    print(
        values_3rd_quartile[['std_Hellinger', 'std_Jensenshannon', 'std_Trace', 'std_Fidelity', 'std_Expectation']])
    print('----------------------------------------')
    print('Threshold: ')
    print(f"Hellinger: {values_3rd_quartile['mean_Hellinger'] + values_3rd_quartile['std_Hellinger'] / np.sqrt(n)}")
    print(f"Jensenshannon: {values_3rd_quartile['mean_Jensenshannon'] + values_3rd_quartile['std_Jensenshannon'] / np.sqrt(n)}")
    print(f"Trace: {values_3rd_quartile['mean_Trace'] + values_3rd_quartile['std_Trace'] / np.sqrt(n)}")
    print(f"Fidelity: {1 - (1 - values_1st_quartile['mean_Fidelity']) + values_1st_quartile['std_Fidelity'] / np.sqrt(n)}")
    print(f"Expectation: {values_3rd_quartile['mean_Expectation'] + values_3rd_quartile['std_Expectation'] / np.sqrt(n)}")
    print('----------------------------------------')


def main():
    # Define a directory for temporary files

    folders = {'Brisbane': '1PON1weLj829TLMRqdgGDx0LJz8rt8FEb',
               'Sherbrooke': '1a2OJ3eaJBVK3pNdneryXEYZF7nGVYiDP',
               'Kyiv': '1LtEq3rt6v2J3xSR88maOHdgreYwuE_4L'}

    service = authenticate_google_drive()

    print('====================== IDEAL THRESHOLDS =========================')
    temp_dir = "results_original_30_runs/results_Ideal"
    os.makedirs(temp_dir, exist_ok=True)
    processed_runs = process_runs(service, folders.get('Brisbane'), False, temp_dir)
    display(processed_runs)

    for folder_name, folder_id in folders.items():
        print(f'====================== NOISY THRESHOLDS FOR {folder_name} =========================')
        temp_dir = "results_original_30_runs/results_" + folder_name
        os.makedirs(temp_dir, exist_ok=True)
        processed_runs = process_runs(service, folder_id, True, temp_dir)
        display(processed_runs)


if __name__ == "__main__":
    main()

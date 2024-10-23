import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files
from distances import fidelityCalc, traceDist, getHellinger, compareChisquare, jensenShannonDivergence


def get_noisy_thresholds(oracle_data):
    column_names = ['Name', 'Input', 'Chisquare', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation']
    # Convert the oracle_data list of dictionaries into a lookup dictionary for faster access
    oracle_lookup = {item['Input']: item for item in oracle_data}

    results = []
    for key, value in oracle_lookup.items():
        chisquare = compareChisquare(value['Ideal_output_distribution'], value['Noisy_output_distribution'])
        hellinger = getHellinger(value['Ideal_output_distribution'], value['Noisy_output_distribution'])
        jensen = jensenShannonDivergence(value['Ideal_output_distribution'], value['Noisy_output_distribution'])
        fidelity = fidelityCalc(value['Ideal_density_matrix'], value['Noisy_density_matrix'])
        trace = traceDist(value['Ideal_density_matrix'], value['Noisy_density_matrix'])
        expectation = abs(value['Ideal_expectation_value']-value['Noisy_expectation_value'])
        # chisquare = 0
        # hellinger = 0
        # jensen = 0
        # fidelity = 0
        # trace = 0
        # expectation = 0
        name = value['Name'].split('/')[-1]
        new_line = {'Name': name, 'Input': value['Input'], 'Chisquare': chisquare, 'Hellinger': hellinger, 'Jensenshannon': jensen, 'Trace': trace, 'Fidelity': fidelity, 'Expectation': expectation}
        results.append(new_line)

    # Convert the results list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results, columns=column_names)
    # mean_values = results_df.iloc[:, 2:].mean()  # Exclude the 'Name' column
    # mean_row = pd.DataFrame([mean_values])
    # mean_row.insert(0, 'Name', name)  # Insert 'Name' column

    return results_df


def process_files(service, origin_id):

    origin_files = get_files(service, origin_id)
    df_total = pd.DataFrame(columns=['Name', 'Input', 'Chisquare', 'Hellinger', 'Jensenshannon', 'Trace', 'Fidelity', 'Expectation'])
    for item in tqdm(origin_files, desc="Checking results..."):
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                pattern = r"indep_qiskit_|_output|.pkl"
                circuit_name = re.sub(pattern, "", filename)
                print(circuit_name)
                oracle_pkl = load_pickle_content(service, file_id)
                if isinstance(oracle_pkl, list):
                        new_df = get_noisy_thresholds(oracle_pkl)
                        df_total = pd.concat([df_total, new_df], ignore_index=True)
                else:
                    print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                    sys.exit(1)
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')
    #print(df_total)
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
def main():
    origin_id = "1p2toGbzc3bAITMic7J1nYccjdkTFpOor"
    service = authenticate_google_drive()
    process_files(service, origin_id)

if __name__ == "__main__":
    main()
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from connectDriveCloud import authenticate_google_drive, load_pickle_content, get_files, get_folders
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
        expectation = abs(value['Ideal_expectation_value'] - value['Noisy_expectation_value'])
        name = value['Name'].split('/')[-1]
        new_line = {'Name': name, 'Input': value['Input'], 'Chisquare': chisquare, 'Hellinger': hellinger,
                    'Jensenshannon': jensen, 'Trace': trace, 'Fidelity': fidelity, 'Expectation': expectation}
        results.append(new_line)

    results_df = pd.DataFrame(results, columns=column_names)

    return results_df


def process_files(service, origin_id):
    origin_files = get_files(service, origin_id)
    df_list = []
    for item in tqdm(origin_files, desc="Checking results..."):
        filename = item['name']
        file_id = item['id']
        if filename.endswith('.pkl'):
            try:
                oracle_pkl = load_pickle_content(service, file_id)
                if isinstance(oracle_pkl, list):
                    new_df = get_noisy_thresholds(oracle_pkl)
                    df_list.append(new_df)
                else:
                    print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                    sys.exit(1)
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')
    df_total = pd.concat(df_list, ignore_index=True)
    # print(df_total)
    return df_total


def main():

    folder_id = '1fo3wbJnzcl_hoVkvpVlCT2Qmwbz7qhpS' # Fake_Brisbane origin_qc folder
    #folder_id = '1ciNw4s_4KndNV0-LuffnQ2o_ynYRLioq' # Fake_Sherbrooke origin_qc folder
    #folder_id = '1ZWvscMpwjw1d8XGSBmwKm3AayUXLYvcK' # Fake_Kyiv origin_qc folder


    service = authenticate_google_drive()
    runs = get_folders(service, folder_id)
    runs_df_list = []
    for run in runs:
        run_id = run['id']
        df_run = process_files(service, run_id)
        runs_df_list.append(df_run)

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

    #print(final_df)
    values = final_df.iloc[:, :-2].median()
    n = len(final_df)  # Number of observations
    print('----------------------------------------')
    print('Mean: ')
    print(values[['mean_Chisquare', 'mean_Hellinger', 'mean_Jensenshannon', 'mean_Trace', 'mean_Fidelity', 'mean_Expectation']])
    print('----------------------------------------')
    print('Standard deviation: ')
    print(values[['std_Chisquare','std_Hellinger','std_Jensenshannon','std_Trace','std_Fidelity','std_Expectation']])
    print('----------------------------------------')
    print('Threshold: ')
    print(f"Chisquare: {values['mean_Chisquare'] + values['std_Chisquare']/ np.sqrt(n)}")
    print(f"Hellinger: {values['mean_Hellinger'] + values['std_Hellinger']/ np.sqrt(n)}")
    print(f"Jensenshannon: {values['mean_Jensenshannon'] + values['std_Jensenshannon']/ np.sqrt(n)}")
    print(f"Trace: {values['mean_Trace'] + values['std_Trace']/ np.sqrt(n)}")
    print(f"Fidelity: {1-(1 - values['mean_Fidelity']) + values['std_Fidelity']/ np.sqrt(n)}")
    print(f"Expectation: {values['mean_Expectation'] + values['std_Expectation']/ np.sqrt(n)}")
    print('----------------------------------------')
    # Display the result
    # final_df.to_csv('runs_exp_results/final_runs_df_marrakesh_max.csv')

if __name__ == "__main__":
    main()

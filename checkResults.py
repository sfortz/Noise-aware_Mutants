import tkinter as tk
from tkinter import filedialog
import pandas as pd

from distances import fidelityCalc, traceDist, getHellinger, compareChisquare


def select_mutants():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select mutants results CSV", filetypes=[("CSV files", "*.csv")])
    return file_paths

def select_oracle():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    oracle_path = filedialog.askopenfilename(title="Select oracle CSV", filetypes=[("CSV files", "*.csv")])
    oracle_df = pd.read_csv(oracle_path)
    return oracle_df


def load_and_merge_files(file_paths):
    # Load the first file into a DataFrame
    if not file_paths:
        return None

    merged_df = pd.read_csv(file_paths[0])

    # Iterate through the remaining files and merge them
    for file_path in file_paths[1:]:
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df


def checkResults(oracle_df, mutants_df):
    column_names = ['Name', 'Input', 'Ideal_chisquare', 'Noisy_chisquare', 'Ideal_hellinger', 'Noisy_hellinger', 'Ideal_trace', 'Noisy_trace', 'Ideal_fidelity', 'Noisy_fidelity','Killed_IC','Killed_NC','Killed_IH','Killed_NH','Killed_IT','Killed_NT','Killed_IF','Killed_NF']
    results_df = pd.DataFrame(columns=column_names)

    for ind, row in mutants_df.iterrows():
        match = oracle_df[oracle_df['Input'] == row.Input]
        if not match.empty:
            killed_IC = False
            killed_NC = False
            killed_IH = False
            killed_NH = False
            killed_IT = False
            killed_NT = False
            killed_IF = False
            killed_NF = False
            index = match.index[0]

            oracle_output = oracle_df.at[index, 'Ideal_output_distribution']
            mutant_output = row.Ideal_output_distribution
            ideal_chisquare = compareChisquare(oracle_output, mutant_output)

            oracle_output = oracle_df.at[index, 'Noisy_output_distribution']
            mutant_output = row.Ideal_output_distribution
            noisy_chisquare = compareChisquare(oracle_output, mutant_output)

            oracle_output = oracle_df.at[index, 'Ideal_output_distribution']
            mutant_output = row.Ideal_output_distribution
            ideal_hellinger = getHellinger(oracle_output, mutant_output)

            oracle_output = oracle_df.at[index, 'Noisy_output_distribution']
            mutant_output = row.Ideal_output_distribution
            noisy_hellinger = getHellinger(oracle_output, mutant_output)

            oracle_output = oracle_df.at[index, 'Ideal_statevector']
            mutant_output = row.Ideal_statevector
            ideal_fidelity = fidelityCalc(oracle_output, mutant_output)
            ideal_trace = traceDist(oracle_output,mutant_output)

            oracle_output = oracle_df.at[index, 'Noisy_statevector']
            mutant_output = row.Noisy_statevector
            noisy_fidelity = fidelityCalc(oracle_output, mutant_output)
            noisy_trace = traceDist(oracle_output, mutant_output)

            tolerance = 1 - 1e-5
            if ideal_fidelity < tolerance:
                killed_IF = True
            if noisy_fidelity < tolerance:
                killed_NF = True

            tolerance = 1e-5
            if ideal_trace > tolerance:
                killed_IT = True
            if noisy_trace > tolerance:
                killed_NT = True

            tolerance = 0.05
            if ideal_hellinger > tolerance:
                killed_IH = True
            if noisy_hellinger > tolerance:
                killed_NH = True

            tolerance = 0.01
            if ideal_chisquare < tolerance:
                killed_IC = True
            if noisy_chisquare < tolerance:
                killed_NC = True

            new_line = {'Name': row.Name, 'Input': row.Input, 'Ideal_chisquare': ideal_chisquare, 'Noisy_chisquare': noisy_chisquare, 'Ideal_hellinger': ideal_hellinger, 'Noisy_hellinger': noisy_hellinger, 'Ideal_trace': ideal_trace, 'Noisy_trace': noisy_trace, 'Ideal_fidelity': ideal_fidelity, 'Noisy_fidelity': noisy_fidelity,'Killed_IC': killed_IC, 'Killed_NC': killed_NC, 'Killed_IH': killed_IH,'Killed_NH': killed_NH,'Killed_IT': killed_IT,'Killed_NT': killed_NT,'Killed_IF': killed_IF,'Killed_NF': killed_NF}
            new_df = pd.DataFrame.from_dict(new_line, orient='index').T
            results_df = pd.concat([results_df, new_df], ignore_index=True)

    return results_df


def main():
    mutants_paths = select_mutants()
    if not mutants_paths:
        print("No files selected.")
        return
    mutants_df = load_and_merge_files(mutants_paths)

    oracle_df = select_oracle()

    results_df = checkResults(oracle_df, mutants_df)
    results_df.to_csv('results_ghz.csv')



if __name__ == "__main__":
    main()

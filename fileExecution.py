import sys
import os
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Assuming you already have setup your IBM quantum account
service = QiskitRuntimeService()

# Get noisy_simulator
ideal_simulator = AerSimulator()

# Get noisy_simulator
noisy_backend = service.backend("ibm_sherbrooke")
noise_model = NoiseModel.from_backend(noisy_backend)
noisy_simulator = AerSimulator(noise_model=noise_model)


def create_inputs(QubitNum):
    inputs = ("",)
    x = 0
    while x < 2 ** QubitNum:
        binariInput = str(bin(x))
        binariInput = binariInput[2:len(binariInput)]
        if len(binariInput) < QubitNum:
            y = len(binariInput)
            tmp = ""
            while y < QubitNum:
                tmp = tmp + str(0)
                y = y + 1
            binariInput = tmp + binariInput
        inputs = inputs + (binariInput,)
        x = x + 1
    return inputs[1:len(inputs)]


def circuit_initialization(qc, input):
    initialization = QuantumCircuit(qc.num_qubits)
    x = 0
    for bit in input:
        if bit == '1':
            initialization.x(x)
        x = x + 1

    qc = initialization.compose(qc)
    return qc


def output_distribution(qc_init):
    # Transpile
    pm_ideal = generate_preset_pass_manager(backend=ideal_simulator, optimization_level=0, seed_transpiler=128)
    pm_noisy = generate_preset_pass_manager(backend=noisy_simulator, optimization_level=0, seed_transpiler=128)
    isa_circuit_ideal = pm_ideal.run(qc_init)
    isa_circuit_noisy = pm_noisy.run(qc_init)

    # Run and get counts
    num_shots = 1024
    job = ideal_simulator.run(isa_circuit_ideal, shots=num_shots, seed=42)
    ideal_counts = job.result().get_counts(isa_circuit_ideal)
    job = noisy_simulator.run(isa_circuit_noisy, shots=num_shots, seed=42)
    noisy_counts = job.result().get_counts(isa_circuit_noisy)

    return ideal_counts, noisy_counts


def final_density_matrix(qc_init):
    qc_init.remove_final_measurements()
    qc_init.save_density_matrix()

    # Transpile
    pm_ideal = generate_preset_pass_manager(backend=ideal_simulator, optimization_level=0, seed_transpiler=128)
    basis_gates = ["h", "cx", "cp", "swap", "barrier", "measure"]
    pm_noisy = generate_preset_pass_manager(backend=noisy_simulator, optimization_level=0, seed_transpiler=128,
                                            basis_gates=basis_gates)
    isa_circuit_ideal = pm_ideal.run(qc_init)
    isa_circuit_noisy = pm_noisy.run(qc_init)

    # Run and get density matrix
    num_shots = 1024
    job = ideal_simulator.run(isa_circuit_ideal, shots=num_shots, seed=42)
    ideal_dm = job.result().data().get('density_matrix')
    job = noisy_simulator.run(isa_circuit_noisy, shots=num_shots, seed=42)
    noisy_dm = job.result().data().get('density_matrix')

    return ideal_dm, noisy_dm


def execute_file(qc, filename):
    df = pd.DataFrame(
        columns=['Name', 'Input', 'Ideal_output_distribution', 'Ideal_density_matrix', 'Noisy_output_distribution',
                 'Noisy_density_matrix'])

    inputs = create_inputs(qc.num_qubits)

    for inp in inputs:
        qc_init = circuit_initialization(qc, inp)
        ideal_out_dist, noisy_out_dist = output_distribution(qc_init)
        ideal_dm, noisy_dm = final_density_matrix(qc_init)

        new_line = {'Name': filename, 'Input': inp, 'Ideal_output_distribution': ideal_out_dist,
                    'Ideal_density_matrix': ideal_dm, 'Noisy_output_distribution': noisy_out_dist,
                    'Noisy_density_matrix': noisy_dm}
        new_df = pd.DataFrame.from_dict(new_line, orient='index').T
        df = pd.concat([df, new_df], ignore_index=True)

    return df


def process_file(filepath, base_input_dir, base_output_dir):
    if not filepath.endswith('.qasm'):
        print(f"Ignoring non-qasm file: {filepath}")
        return

    # Create the relative path for the output directory
    relative_path = os.path.relpath(filepath, base_input_dir)
    outputfile = relative_path.replace('.qasm', '_output.csv')
    output_path = os.path.join(base_output_dir, outputfile)

    print(f"Executing: {filepath}")
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"Output file already exists for {filepath}, skipping.")
        return

    try:
        # Execute the file and get the result as a DataFrame
        qc = QuantumCircuit.from_qasm_file(filepath)
        result_df = execute_file(qc, filepath)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the result DataFrame as a CSV file
        result_df.to_csv(output_path, index=False)
        print(f"Result saved as {output_path}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def main():
    # Check if a path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python fileExecution.py <file_or_directory_path>")
        sys.exit(1)

    input_dir = sys.argv[1]  # Get the input dir from command-line arguments
    base_input_dir = 'experiment/' + input_dir  # Define the base input directory
    base_output_dir = 'exec/test/' + input_dir  # Define the base output directory

    # Check if the path is a directory or a file
    if os.path.isdir(base_input_dir):
        # Process all .qasm files in the directory and subdirectories
        for root, _, files in os.walk(base_input_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                process_file(filepath, base_input_dir, base_output_dir)
    elif os.path.isfile(base_input_dir):
        # Process the single file
        process_file(base_input_dir, os.path.dirname(base_input_dir), os.path.dirname(base_output_dir))
    else:
        print("Error: The provided path is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()

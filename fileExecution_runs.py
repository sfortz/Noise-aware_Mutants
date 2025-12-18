import sys
import os
import pickle

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from noisemodel import load_noise_model

basis_gates = ["ecr", "id", "rz", "sx", "x"] #Sherbrooke, brisbane, kyiv
noise_model = load_noise_model("ibm_brisbane_noise_model.pkl")
noisy_simulator = AerSimulator(noise_model=noise_model)
ideal_simulator = AerSimulator()


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
    pm_ideal = generate_preset_pass_manager(backend=ideal_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128)
    pm_noisy = generate_preset_pass_manager(backend=noisy_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128)
    isa_circuit_ideal = pm_ideal.run(qc_init)
    isa_circuit_noisy = pm_noisy.run(qc_init)

    # Run and get counts
    num_shots = 10000
    job = ideal_simulator.run(isa_circuit_ideal, shots=num_shots)#, seed=42)
    ideal_counts = job.result().get_counts(isa_circuit_ideal)
    job = noisy_simulator.run(isa_circuit_noisy, shots=num_shots)#, seed=42)
    noisy_counts = job.result().get_counts(isa_circuit_noisy)

    return ideal_counts, noisy_counts

def expectation_values(qc_init):
    # Step 2: Define observable
    observable = SparsePauliOp("Z" * qc_init.num_qubits)
    # Transpile
    pm_ideal = generate_preset_pass_manager(backend=ideal_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128)
    pm_noisy = generate_preset_pass_manager(backend=noisy_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128)
    isa_circuit_ideal = pm_ideal.run(qc_init)
    isa_circuit_noisy = pm_noisy.run(qc_init)

    ideal_estimator = Estimator(ideal_simulator)#, options={'seed_simulator': 42})
    noisy_estimator = Estimator(noisy_simulator)#, options={'seed_simulator': 42})

    job_ideal = ideal_estimator.run([(isa_circuit_ideal, observable)])
    result_ideal = job_ideal.result()[0]
    ideal_expectation = result_ideal.data.evs.item()

    job_noisy = noisy_estimator.run([(isa_circuit_noisy, observable)])
    result_noisy = job_noisy.result()[0]
    noisy_expectation = result_noisy.data.evs.item()

    return ideal_expectation, noisy_expectation


def final_density_matrix(qc_init):
    qc_init.remove_final_measurements()
    qc_init.save_density_matrix()

    # Transpile
    pm_ideal = generate_preset_pass_manager(backend=ideal_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128)
    pm_noisy = generate_preset_pass_manager(backend=noisy_simulator, optimization_level=0, basis_gates=basis_gates)#, seed_transpiler=128,
                                            
    isa_circuit_ideal = pm_ideal.run(qc_init)
    isa_circuit_noisy = pm_noisy.run(qc_init)

    # Run and get density matrix
    num_shots = 10000
    job = ideal_simulator.run(isa_circuit_ideal, shots=num_shots)#, seed=42)
    ideal_dm = job.result().data().get('density_matrix')
    job = noisy_simulator.run(isa_circuit_noisy, shots=num_shots)#, seed=42)
    noisy_dm = job.result().data().get('density_matrix')

    return ideal_dm.data, noisy_dm.data


def get_inputs(pure_state, num_qubits):

    if pure_state:
        folder = f'data/pure_state_inputs/inputs_{num_qubits}_qubits'
    else:
        folder = f'data/quratest_inputs/inputs_{num_qubits}_qubits'

    inputs = []
    for filename in os.listdir(folder):
        # Check if the filename has the .qasm extension
        if filename.endswith('.qasm'):
            input = os.path.join(folder, filename)
            input_qc = QuantumCircuit.from_qasm_file(input)
            inputs.append(input_qc)

    return inputs


def process_file(filepath, base_input_dir, base_output_dir):
    if not filepath.endswith('.qasm'):
        print(f"Ignoring non-qasm file: {filepath}")
        return

    # Create the relative path for the output directory
    relative_path = os.path.relpath(filepath, base_input_dir)
    outputfile = relative_path.replace('.qasm', '_output.pkl')
    output_path = os.path.join(base_output_dir, outputfile)

    print(f"Executing: {filepath}")
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"Output file already exists for {filepath}, skipping.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Execute the file and get the result as a JSON
        qc = QuantumCircuit.from_qasm_file(filepath)
        pure_state_inputs = get_inputs(True, qc.num_qubits)
        quratest_inputs = get_inputs(False, qc.num_qubits)

        results = []

        for index, (ps_input, qt_input) in enumerate(zip(pure_state_inputs, quratest_inputs)):
            qc_init_ps = ps_input.copy()
            qc_init_ps = qc_init_ps.compose(qc)
            ps_ideal_out_dist, ps_noisy_out_dist = output_distribution(qc_init_ps)
            ps_ideal_exp_val, ps_noisy_exp_val = expectation_values(qc_init_ps)
            ps_ideal_dm, ps_noisy_dm = final_density_matrix(qc_init_ps)

            # Prepare PureState data row
            pure_state_row = {
                'Name': filepath,
                'Input': f'PureState_{index}',
                'Ideal_output_distribution': ps_ideal_out_dist,
                'Ideal_expectation_value': ps_ideal_exp_val,
                'Ideal_density_matrix': ps_ideal_dm,
                'Noisy_output_distribution': ps_noisy_out_dist,
                'Noisy_expectation_value': ps_noisy_exp_val,
                'Noisy_density_matrix': ps_noisy_dm
            }
            results.append(pure_state_row)

            qc_init_qt = qt_input.copy()
            qc_init_qt = qc_init_qt.compose(qc)
            qt_ideal_out_dist, qt_noisy_out_dist = output_distribution(qc_init_qt)
            qt_ideal_exp_val, qt_noisy_exp_val = expectation_values(qc_init_qt)
            qt_ideal_dm, qt_noisy_dm = final_density_matrix(qc_init_qt)

            # Prepare Quratest data row
            quratest_row = {
                'Name': filepath,
                'Input': f'Quratest_{index}',
                'Ideal_output_distribution': qt_ideal_out_dist,
                'Ideal_expectation_value': qt_ideal_exp_val,
                'Ideal_density_matrix': qt_ideal_dm,
                'Noisy_output_distribution': qt_noisy_out_dist,
                'Noisy_expectation_value': qt_noisy_exp_val,
                'Noisy_density_matrix': qt_noisy_dm
            }
            results.append(quratest_row)

        # Save results using Pickle
        with open(output_path, 'wb') as file:
            pickle.dump(results, file)

        print(f"Output saved to {output_path}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def main():
    # Check if a path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python fileExecution.py <file_or_directory_path>")
        sys.exit(1)

    input_dir = sys.argv[1]  # Get the input dir from command-line arguments
    #base_input_dir = 'data/' + input_dir  # Define the base input directory
    for x in range(10):
        #new_input_dir = input_dir.replace('data/equiv_mutants/','')
        #base_output_dir = f'exec_fake_kyiv/equiv_qc/run_{x}/{new_input_dir}/'  # Define the base output directory
        base_output_dir = f'exec_fake_brisbane/origin_qc/run_{x}/'

        # Check if the path is a directory or a file
        if os.path.isdir(input_dir):
            # Process all .qasm files in the directory and subdirectories
            for root, _, files in os.walk(input_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    process_file(filepath, input_dir, base_output_dir)
        elif os.path.isfile(input_dir):
            # Process the single file
            process_file(input_dir, os.path.dirname(input_dir), os.path.dirname(base_output_dir))
        else:
            print("Error: The provided path is neither a file nor a directory.")
            sys.exit(1)


if __name__ == "__main__":
    main()

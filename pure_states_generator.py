import os
import sys

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

backend = AerSimulator()


def generate_random_pure_state(n_qbits):
    # Create a Quantum Circuit with n qubits and n classical bits
    qc = QuantumCircuit(n_qbits, n_qbits)

    # Apply a Hadamard gate to each qubit to create a superposition
    for qubit in range(n_qbits):
        qc.h(qubit)

    # Measure each qubit and store the result in the corresponding classical bit
    qc.measure(range(n_qbits), range(n_qbits))

    # Transpile
    pm_ideal = generate_preset_pass_manager(backend=backend, optimization_level=0, seed_transpiler=128)
    isa_circuit = pm_ideal.run(qc)

    # Run and get counts
    job = backend.run(isa_circuit, shots=1, seed=42)
    counts = job.result().get_counts(isa_circuit)

    # Get the key of the 1-shot measurement
    measurement_result = list(counts.keys())[0]

    return measurement_result


def generate_qasm(binary_string, file_path):
    num_qubits = len(binary_string)
    qasm_code = [
        "OPENQASM 2.0;\n",
        "include \"qelib1.inc\";\n",
        f"qreg q0[{num_qubits}];\n",
        f"creg c0[{num_qubits}];\n"
    ]

    # Apply X gates based on the binary string
    for i, bit in enumerate(binary_string):
        if bit == '1':
            qasm_code.append(f"x q0[{i}];\n")

    # Write the QASM code to the file
    with open(file_path, "w") as file:
        file.write(''.join(qasm_code))


def pure_states_generator(n_qbits, n_inputs):
    if n_inputs > 2 ** n_qbits:
        print("It is impossible to generate more than 2^n_qubits different pure states.")
        sys.exit(1)

    dir_path = "data/pure_state_inputs/inputs_" + str(n_qbits) + "_qubits/"
    os.makedirs(dir_path, exist_ok=True)
    random_pure_states = []
    num_files = 0

    while len(random_pure_states) < n_inputs:
        random_input = generate_random_pure_state(n_qbits)
        if random_input not in random_pure_states:
            random_pure_states.append(random_input)
            file_name = "input_" + str(num_files)
            file_path = dir_path + file_name + ".qasm"
            generate_qasm(random_input, file_path)
            num_files = num_files + 1

    return random_pure_states


if __name__ == "__main__":

    for n_qubits in range(2, 11):
        max_pure_states = 2 ** n_qubits
        n_inputs = int(max_pure_states / 2)

        if n_inputs > 10:
            n_inputs = 10

        pure_states_generator(n_qubits, n_inputs)

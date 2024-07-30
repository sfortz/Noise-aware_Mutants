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


def pure_states_generator(n_qbits, n_input):
    random_pure_states = []

    while len(random_pure_states) < n_input:
        random_input = generate_random_pure_state(n_qbits)
        if random_input not in random_pure_states:
            random_pure_states.append(random_input)

    return random_pure_states

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit_ibm_runtime import QiskitRuntimeService


def get_noise_model_generic_qubits(calibration_df, supported_gates):
    columns = ["T1", "T2", "assigment_error", "prob01", "prob10", "Readout length (ns)", "time"] + supported_gates
    values = calibration_df[columns].median()

    # Create an empty noise model
    noise_model = NoiseModel(basis_gates=supported_gates)

    for gate in supported_gates:
        if gate == 'ecr' or gate == 'cz' or gate == 'rzz':
            depo_error = depolarizing_error(values[gate], 1).expand(depolarizing_error(values[gate], 1))
            thermal_error = thermal_relaxation_error(values['T1'], values['T2'], values['time']).expand(thermal_relaxation_error(values['T1'], values['T2'], values['time']))
            #error = error.compose(pauli_error([('X', values['assigment_error']), ('I', 1 - values['assigment_error'])]))
            noise_model.add_all_qubit_quantum_error(depo_error, [gate])
            noise_model.add_all_qubit_quantum_error(thermal_error, [gate])

        else:
            depo_error = depolarizing_error(values[gate], 1)
            thermal_error = thermal_relaxation_error(values['T1'], values['T2'], values['time'])
            #error = error.compose(pauli_error([('X', values['assigment_error']), ('I', 1 - values['assigment_error'])]))
            noise_model.add_all_qubit_quantum_error(depo_error, [gate])
            noise_model.add_all_qubit_quantum_error(thermal_error, [gate])

    readout_error = ReadoutError([[1 - values['prob10'], values['prob10']], [values['prob01'], 1 - values['prob01']]])
    noise_model.add_all_qubit_readout_error(readout_error)
    # print(noise_model)
    return noise_model

def get_noise_model_individual_qubits(calibration_df, supported_gates):
    columns = ["T1", "T2", "assigment_error", "prob01", "prob10", "Readout length (ns)", "time"] + supported_gates
    values = calibration_df[columns].median()

    noise_model = NoiseModel(basis_gates=supported_gates)

    # Iterate through qubits
    for index, row in calibration_df.iterrows():
        for gate in supported_gates:
            if gate != 'ecr' or gate != 'cz' or gate != 'rzz':
                depo_error = depolarizing_error(row[gate], 1)
                thermal_error = thermal_relaxation_error(row['T1'], row['T2'], row['time'])
                noise_model.add_quantum_error(depo_error, [gate], [row['Qubit']])
                noise_model.add_quantum_error(thermal_error, [gate], [row['Qubit']])

        readout_error = ReadoutError([[1 - row['prob10'], row['prob10']], [row['prob01'], 1 - row['prob01']]])
        noise_model.add_readout_error(readout_error, [row['Qubit']])

    if 'ecr' in supported_gates:
        depo_error = depolarizing_error(values['ecr'], 1).expand(depolarizing_error(values['ecr'], 1))
        thermal_error = thermal_relaxation_error(values['T1'], values['T2'], values['time']).expand(thermal_relaxation_error(values['T1'], values['T2'], values['time']))
        noise_model.add_all_qubit_quantum_error(depo_error, ['ecr'])
        noise_model.add_all_qubit_quantum_error(thermal_error, ['ecr'])

    else:
        depo_error = depolarizing_error(values['cz'], 1).expand(depolarizing_error(values['cz'], 1))
        thermal_error = thermal_relaxation_error(values['T1'], values['T2'], values['time']).expand(
            thermal_relaxation_error(values['T1'], values['T2'], values['time']))
        noise_model.add_all_qubit_quantum_error(depo_error, ['cz'])
        noise_model.add_all_qubit_quantum_error(thermal_error, ['cz'])

        depo_error = depolarizing_error(values['rzz'], 1).expand(depolarizing_error(values['rzz'], 1))
        thermal_error = thermal_relaxation_error(values['T1'], values['T2'], values['time']).expand(
            thermal_relaxation_error(values['T1'], values['T2'], values['time']))
        noise_model.add_all_qubit_quantum_error(depo_error, ['rzz'])
        noise_model.add_all_qubit_quantum_error(thermal_error, ['rzz'])

    return noise_model

def get_noise_model_without_noise(calibration_df, supported_gates):
    # Create an empty noise model
    noise_model = NoiseModel(basis_gates=supported_gates)

    return noise_model

def get_IBM_backend_noise_model(model_name):
    # Save an IBM Quantum account.
    service = QiskitRuntimeService(channel="ibm_quantum",token="67ae1be6477a413c62411bc6120a39ab6244f3037c3439013e07f9d2509efba39d3329f7adee6622cc6136cce7cf0dfffbeae95433f9d613a6c84b7de61e6913")

    noisy_backend = service.backend(model_name)
    noise_model = NoiseModel.from_backend(noisy_backend)

    return noise_model
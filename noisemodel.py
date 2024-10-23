import numpy as np
import pandas as pd
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError, pauli_error

def get_noise_model(calibration_df, supported_gates):
    values = calibration_df[["T1", "T2", "assigment_error", "prob01", "prob10", "Readout length (ns)", "id", "rz", "sx", "x", "ecr_1", "ecr_2", "ecr_3", "time"]].median()
    # Create an empty noise model
    noise_model = NoiseModel(basis_gates=supported_gates)

    for gate in supported_gates:
        if gate == 'ecr':
            error = depolarizing_error(values[gate+'_1'], 1).expand(depolarizing_error(values[gate+'_2'], 1))
            error = error.compose(thermal_relaxation_error(values['T1'], values['T2'], values['time']).expand(thermal_relaxation_error(values['T1'], values['T2'], values['time'])))
            #error = error.compose(pauli_error([('X', values['assigment_error']), ('I', 1 - values['assigment_error'])]))
            noise_model.add_all_qubit_quantum_error(error, [gate])
        else:
            error = depolarizing_error(values[gate], 1)
            error = error.compose(thermal_relaxation_error(values['T1'], values['T2'], values['time']))
            #error = error.compose(pauli_error([('X', values['assigment_error']), ('I', 1 - values['assigment_error'])]))
            noise_model.add_all_qubit_quantum_error(error, [gate])

    readout_error = ReadoutError([[1 - values['prob10'], values['prob10']], [values['prob01'], 1 - values['prob01']]])
    noise_model.add_all_qubit_readout_error(readout_error)
    # print(noise_model)
    return noise_model

# calibration_df = pd.read_csv("calibration_values/ibm_sherbrooke_calibrations_2024-10-23T09_03_10Z.csv")
# gates = ['ecr', 'id', 'rz', 'sx', 'x']
# get_noise_model(calibration_df, gates)
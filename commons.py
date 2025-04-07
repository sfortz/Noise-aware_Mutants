import os

thresholds = ['A', 'N', 'M', 'I']  #'0.1','0.5','0.8','A',
hardware = ["kyiv", "brisbane", "sherbrooke"]
mutant_types = ["equiv", "normal", "balanced"]
output_type = {'ae': 'Dominant', 'qpeexact': 'Dominant', 'vqe': 'Dominant', 'qft': 'Diverse', 'qftentangled': 'Diverse',
               'wstate': 'Diverse'}

metrics = {'H': 'hellinger', 'J': 'jensenshannon', 'T': 'trace', 'F': 'fidelity', 'E': 'expectation'}
metric_names=('Hellinger', 'Jensen-shannon', 'Trace', 'Fidelity', 'Expectation Values')

table_data = {
    "Output_type": ["Dominant", "Diverse"],
    "Input_type": ["PureState", "Quratest"],
    "Operator": ["Add", "Remove", "Replace"],
    "Gate_type": ["Single_qubit", "Multi_qubit"],
    "Relative_position": ["beginning", "pre_middle", "middle", "post_middle", "end"]
}

tolerance_values_ideal = {
    'fidelity': 1 - 1e-14,
    'trace': 1e-13,
    'hellinger': 0.05999462573410876,
    'jensenshannon': 0.0573756781781595,
    'expectation': 0.01341994679081911
}


def getModelTolerance(model):
    if model == 'brisbane':
        tolerance_values_noisy = {
            'fidelity': 0.844560981448548,
            'trace': 0.06156115753169809,
            'hellinger': 0.30945013336752103,
            'jensenshannon': 0.2648394335745957,
            'expectation': 0.08662027935711168
        }

    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 0.7091486297474972,
            'trace': 0.10230744393882127,
            'hellinger': 0.31828796082696786,
            'jensenshannon': 0.2773210504137734,
            'expectation': 0.07520077028113412
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 0.842908980932802,
            'trace': 0.0684393918944923,
            'hellinger': 0.2630386511490515,
            'jensenshannon': 0.22471852013465599,
            'expectation': 0.04963636739920271
        }

    else:
        tolerance_values_noisy = {}

    return tolerance_values_noisy


def get_tolerance_values(model, threshold):
    tolerance_values_ideal = {
        'fidelity': 1 - 1e-14,
        'trace': 1e-13,
        'hellinger': 0.05999462573410876,
        'jensenshannon': 0.0573756781781595,
        'expectation': 0.01341994679081911
    }

    # Define tolerance values
    if threshold == 'I':
        tolerance_values = tolerance_values_ideal
    elif threshold == 'N':
        tolerance_values = getModelTolerance(model)
    elif threshold == 'M':
        tolerance_values = {
            'fidelity': 0.922280490724269,
            'trace': 0.030780578765899045,
            'hellinger': 0.16151663844158012,
            'jensenshannon': 0.14104709915640773,
            'expectation': 0.03152815709501091
        }

    elif threshold == 'A':
        tolerance_values = {
            'fidelity': 0.6314291204717761,
            'trace': 0.1330880227046203,
            'hellinger': 0.4198099735344393,
            'jensenshannon': 0.3609924713920217,
            'expectation': 0.10472848966130349
        }

    return tolerance_values


# Helper function to set up layout and save image
def setup_layout_and_save(fig, title, folder_name, file_name, yaxis_range=None):
    fig.update_layout(
        title_text=title,
        height=400,
        width=2000,
        showlegend=True,
        yaxis_range=yaxis_range,  # Set y-axis range if provided
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    os.makedirs(folder_name, exist_ok=True)
    fig.write_image(f"{folder_name}/{file_name}.png")  # engine='orca')


type_dict = {
    "gates": int,
    "depth": int,
    "singlequbit_gates": int,
    "multiqubit_gates": int,
    "Input": str,
    "Input_type": str,
    "Algorithm": str,
    "Qubits_number": int,
    "Operator": str,
    "Gate": str,
    "Position": int,
    "Qubits": int,
    "Gate_type": str,
    "Relative_position": str,
    "Output_type": str,
    "hardware": str,
    "threshold": str,
    "metric": str,
    "ideal_distance": float,
    "noisy_distance": float,
    "ideal_label": bool,
    "noisy_label": bool,
    "correctness": bool
}
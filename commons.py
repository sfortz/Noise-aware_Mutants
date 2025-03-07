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
    'hellinger': 0.04167558253647273,
    'jensenshannon': 0.04046209568627424,
    'expectation': 0.012071688887134691
}


def getModelTolerance(model):
    if model == 'brisbane':
        tolerance_values_noisy = {
            'fidelity': 0.88969,
            'trace': 0.04414,
            'hellinger': 0.17512,
            'jensenshannon': 0.16315,
            'expectation': 0.02152
        }

    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 0.83609,
            'trace': 0.06454,
            'hellinger': 0.21111,
            'jensenshannon': 0.19657,
            'expectation': 0.02231
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 0.88717,
            'trace': 0.04599,
            'hellinger': 0.14851,
            'jensenshannon': 0.13839,
            'expectation': 0.01720
        }

    else:
        tolerance_values_noisy = {}

    return tolerance_values_noisy


def get_tolerance_values(model, threshold):
    tolerance_values_ideal = {
        'fidelity': 1 - 1e-14,
        'trace': 1e-13,
        'hellinger': 0.04168,
        'jensenshannon': 0.04046,
        'expectation': 0.01207
    }

    # Define tolerance values
    if threshold == 'I':
        tolerance_values = tolerance_values_ideal
    elif threshold == 'N':
        tolerance_values = getModelTolerance(model)
    elif threshold == 'M':
        tolerance_values = {
            'fidelity': 0.94485,
            'trace': 0.02207,
            'hellinger': 0.09509,
            'jensenshannon': 0.08943,
            'expectation': 0.01464
        }

    elif threshold == 'A':
        tolerance_values = {
            'fidelity': 0.78094,
            'trace': 0.08661,
            'hellinger': 0.26453,
            'jensenshannon': 0.24553,
            'expectation': 0.02487
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
    "true_label": bool,
    "predicted_label": bool,
    "correctness": bool
}
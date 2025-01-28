import os

thresholds = ['I', 'N']  #'0.1','0.5','0.8','A',
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
            'fidelity': 0.889694852626923,
            'trace': 0.04413922158357281,
            'hellinger': 0.17511932212075987,
            'jensenshannon': 0.16314623886821963,
            'expectation': 0.021517498392612106
        }

    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 0.8360876866656799,
            'trace': 0.06454239845481724,
            'hellinger': 0.21111207949813202,
            'jensenshannon': 0.19656798430087188,
            'expectation': 0.022305089004287383
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 0.8871746464728272,
            'trace': 0.04598758133101792,
            'hellinger': 0.14851063146748153,
            'jensenshannon': 0.13839484544489133,
            'expectation': 0.017200531608979386
        }

    else:
        tolerance_values_noisy = {}

    return tolerance_values_noisy


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
    "distance": float,
    "true_label": bool,
    "predicted_label": bool,
    "correctness": bool
}
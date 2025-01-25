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
    'hellinger': 0.04178952039843151,
    'jensenshannon': 0.04058294316372258,
    'expectation': 0.011574442770798709
}


def getModelTolerance(model):
    if model == 'brisbane':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9118523593799848,
            'trace': 0.034415871303828394,
            'hellinger': 0.2130665216337895,
            'jensenshannon': 0.19154709011416926,
            'expectation': 0.019239830427815005
        }

    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.7606042378746769,
            'trace': 0.09087481763504617,
            'hellinger': 0.2835557485569176,
            'jensenshannon': 0.26311290831221235,
            'expectation': 0.032351843338062784
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 1 - 0.9024474384739689,
            'trace': 0.04045065953149715,
            'hellinger': 0.19741965571741749,
            'jensenshannon': 0.1836415632471504,
            'expectation': 0.0239542540495735
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
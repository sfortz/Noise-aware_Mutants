import os

thresholds = ['I', 'M', 'N', 'A']  #'0.1','0.5','0.8','A',
hardware = ["kyiv", "brisbane", "sherbrooke"]
mutant_types = ["equiv", "normal", "balanced"]
output_type = {'ae': 'Dominant', 'qpeexact': 'Dominant', 'vqe': 'Dominant', 'qft': 'Diverse', 'qftentangled': 'Diverse',
               'wstate': 'Diverse'}

#metrics = {'H': 'hellinger', 'J': 'jensenshannon', 'T': 'trace', 'F': 'fidelity', 'E': 'expectation'}
#metric_names = {'H': 'Hellinger', 'J': 'Jensen-Shannon', 'T': 'Trace', 'F': 'Fidelity', 'E': 'Expectation Values'}
metrics = {'T': 'trace', 'F': 'fidelity', 'H': 'hellinger', 'J': 'jensenshannon', 'E': 'expectation'}
metric_names = {'T': 'Trace', 'F': 'Fidelity', 'H': 'Hellinger', 'J': 'Jensen-Shannon', 'E': 'Expectation Values'}

scores = ["Accuracy", "F1 Score", "Precision", "Recall"]

hardware_names = {
    'ideal': 'Noiseless',
    'kyiv': 'Kyiv noise model',
    'brisbane': 'Brisbane noise model',
    'sherbrooke': 'Sherbrooke noise model'
}

color_map = {
    'Noiseless': '#1f77b4',       # Blue
    'Kyiv noise model': '#a1d99b',      # Light green
    'Brisbane noise model': '#74c476',      # Medium green
    'Sherbrooke noise model': '#238b45'       # Dark green
}

table_data = {
    "Output_type": ["Dominant", "Diverse"],
    "Input_type": ["PureState", "Quratest"],
    "Operator": ["Add", "Remove", "Replace"],
    "Gate_type": ["Single-qubit", "Multi-qubit"],
    "Relative_position": ["Beginning", "Pre middle", "Middle", "Post middle", "End"]
}

def _getModelTolerance(model):
    if model == 'brisbane':
        tolerance_values_noisy = {
            'fidelity': 0.8241670074950497,
            'trace': 0.07080045257375521,
            'hellinger': 0.36616035976726463,
            'jensenshannon': 0.30976931193780083,
            'expectation': 0.27260619447389095
        }
    elif model == 'sherbrooke':
        tolerance_values_noisy = {
            'fidelity': 0.6614564604336312,
            'trace': 0.14328038153045655,
            'hellinger': 0.4275663427429035,
            'jensenshannon': 0.3636798118443935,
            'expectation': 0.2556021335991067
        }
    elif model == 'kyiv':
        tolerance_values_noisy = {
            'fidelity': 0.8048073479158889,
            'trace': 0.08959077991201495,
            'hellinger': 0.32652044200901165,
            'jensenshannon': 0.27610690639943464,
            'expectation': 0.1783660382226895
        }
    else:
        tolerance_values_noisy = {}

    return tolerance_values_noisy


def get_tolerance_values(threshold, model=None):

    # Define tolerance values
    if threshold == 'I':
        tolerance_values = {
            'fidelity': 1 - 1e-14,
            'trace': 1e-13,
            'hellinger': 0.06561412848403149,
            'jensenshannon': 0.06045268171304313,
            'expectation': 0.014284606173994616
    }
    elif threshold == 'N':
        if model == None:
            raise ValueError("For threshold 'N', the 'model' parameter must be specified.")
        tolerance_values = _getModelTolerance(model)
    elif threshold == 'M':
        tolerance_values = {
            'fidelity': 0.9120835037475199,
            'trace': 0.03540022628692761,
            'hellinger': 0.19606728524652156,
            'jensenshannon': 0.16827979405623888,
            'expectation': 0.09632532219834206
        }
    elif threshold == 'A':
        tolerance_values = {
            'fidelity': 0.5735399641811609,
            'trace': 0.17868060781728415,
            'hellinger': 0.5580194995053935,
            'jensenshannon': 0.47150692418758927,
            'expectation': 0.3546469104982384
        }
    else:
        raise ValueError(f"Invalid threshold: {threshold}. Must be one of {thresholds}")

    return tolerance_values


# Helper function to set up layout and save image
def setup_layout_and_save(fig, folder_name, file_name, yaxis_range=None, height=400, width=2000):
    fig.update_xaxes(showgrid=False, zeroline=True)
    fig.update_yaxes(showgrid=False, zeroline=True)
    fig.update_layout(
        #title_text=title,
        height=height,
        width=width,
        showlegend=True,
        yaxis_range=yaxis_range,  # Set y-axis range if provided
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        #font=dict(color="White", size=25)
        font=dict(color="Black", size=25)
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
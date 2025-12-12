# Empirical Evaluation On Noise Aware Quantum Mutant Detection

This repository contains the full workflow for our empirical evaluation, including mutant generation, execution, distance computation, thresholding, and analysis.

An overview of the workflow is presented in the experiment flow diagram below.

![Experiment Workflow](./data/OveralExperimentFlow.png)

## Repository Structure
- `data/` — Raw and processed execution data.
- `results/`, `results_noise_analysis/`, `results_original_30_runs/` — Execution outputs, distance files, and analysis results.
- `noise_models_*/` — Pre-generated noise model configurations.
- Python modules and notebooks for each phase of the workflow.

## Workflow Overview

### A. Mutant Generation and Selection
Scripts used to generate mutants for each CUT and select a representative subset.

**Relevant files:**
- `selectMutants.ipynb`
- `checkMutation.py`
- `pure_states_generator.py`

### B. Execution
Execution of all CUTs and mutants across input configurations and noise profiles.  
This step generates the bulk of the data later stored under `results*/`.

**Relevant files:**
- `fileExecution_runs.py`
- `checkNoise.py`
- `noisemodel.py`
- `connectDriveCloud.py` (optional cloud storage integration)

### C. Distance Measurements
Computation of distance metrics from all execution traces and export to CSV.

**Relevant files:**
- `distances.py`
- `compute_ideal_noisy_thresholds.py`
- `compute_middle_above_thresholds.py`
- `approximate_ideal_thresholds.ipynb`

### D. Threshold Application
Applying predefined thresholds to the computed distances to determine mutant detection outcomes.

**Relevant files:**
- `best_threshold.ipynb`
- `compute_ideal_noisy_thresholds.py`
- `compute_middle_above_thresholds.py`

### E. Analysis
Aggregation of detection results and analysis addressing the research questions.

**Relevant files:**
- `analize_results_mutation.ipynb`
- `analize_results_noise.ipynb`
- `statistics.ipynb`
- `visu_RQ0.ipynb`
- `visu_RQ1.ipynb`
- `visu_RQ2.ipynb`

## Requirements
- Python 3.9+
- Required packages listed in `requirements.txt`

## Citation
If you use this workflow or dataset, please cite our work (citation details here).

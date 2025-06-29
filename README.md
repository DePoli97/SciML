# SciML - Monodomain Equation Solver

This project implements two approaches to solve the monodomain equation for cardiac electrophysiology:

1. A Finite Element Method (FEM) solver
2. A Physics-Informed Neural Network (PINN) solver

The system simulates the propagation of electrical signals in cardiac tissue with different diffusivity scenarios.

## Project Structure

```
├── src/
│   ├── python/         # Python solvers and utilities
│   │   ├── main.py     # Command-line interface
│   │   ├── fem_solver.py   # FEM implementation
│   │   ├── pinn_solver.py  # PINN implementation
│   │   └── plotting.py     # Visualization utilities
│   └── matlab/         # MATLAB reference implementation
├── models/             # PINN model storage
│   └── pinn_model_*    # Each trained model with configs and history
├── assets/             # Output videos and frames
│   ├── fem/            # FEM simulation outputs
│   └── pinn/           # PINN simulation outputs
├── environment.yml     # Conda environment specification
└── README.md           # This file
```

## Installation

Create and activate a conda environment with the required dependencies:

```bash
conda env create -f environment.yml
conda activate sciml-env
```

## Usage

The project can be run from the command line with the following commands:

### Run FEM Simulations

```bash
python -m src.python.main fem --case [high|normal|low|all]
```

- `all`: Run all three cases (default)
- `high`: High diffusivity case (10x normal)
- `normal`: Normal diffusivity case
- `low`: Low diffusivity case (0.1x normal)

### Train a PINN Model

```bash
python -m src.python.main pinn-train [--model-name NAME]
```

- `--model-name`: Optional name for the model directory (default: auto-generated timestamp)

Each training run creates a new directory in `models/` containing:
- `model_weights.pth`: The trained model weights
- `training_loss.png`: Plot of the loss curves during training
- `pinn_solver_script.py`: Copy of the PINN script used for reproducibility

### Generate Predictions with PINN

```bash
python -m src.python.main pinn-predict --case [high|normal|low|all] [--model-name NAME]
```

- `--case`: Diffusivity case to simulate
- `--model-name`: Name of the model to use (default: uses the latest model)

## Project Parameters

The main physical and simulation parameters are defined at the top of `main.py`:

- **Physical parameters**:
  - `SIGMA_H`: Baseline diffusivity coefficient
  - `A`, `FR`, `FT`, `FD`: Reaction parameters for the model

- **Simulation parameters**:
  - `T`: Total simulation time
  - `DT`: Time step size
  - `NVX`, `NVY`: Number of grid points in each dimension

## Workflow Example

A typical workflow might look like:

1. Run the FEM simulation for reference:
   ```
   python -m src.python.main fem --case all
   ```

2. Train a PINN model:
   ```
   python -m src.python.main pinn-train --model-name my_first_model
   ```

3. Generate predictions with the trained model:
   ```
   python -m src.python.main pinn-predict --case all --model-name my_first_model
   ```

4. Compare the results in `assets/fem/` and `assets/pinn/`

## Experiment Management

The system automatically tracks experiments by:
- Saving each PINN model with a timestamp or custom name
- Storing model weights, loss plot, and the exact solver code version
- Organizing outputs in a consistent directory structure

For better experiment tracking, consider using fully descriptive model names:
```
python -m src.python.main pinn-train --model-name reduced_neurons_tanh_activation
```

## Extending the Project

To extend this project with new features:
1. For new PINN architectures, modify the `PINNSolver` class in `pinn_solver.py` **WITHOUT** modifying the class constructor
2. For different physical parameters, adjust the constants in `main.py`
3. For new visualization approaches, add functions to `plotting.py`
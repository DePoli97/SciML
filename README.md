# SciML - Monodomain Equation Solver

This project implements multiple approaches to solve the monodomain equation for cardiac electrophysiology:

1. A Finite Element Method (FEM) solver
2. A Physics-Informed Neural Network (PINN) solver
3. A Convolutional Neural Network (CNN) solver
4. A DeepRitz solver (variational physics-informed neural networks)

The system simulates the propagation of electrical signals in cardiac tissue with different diffusivity scenarios.

## Project Structure

```
├── src/
│   ├── python/         # Python solvers and utilities
│   │   ├── main.py     # Command-line interface
│   │   ├── fem_solver.py   # FEM implementation
│   │   ├── pinn_solver.py  # PINN implementation
│   │   ├── cnn_solver.py   # CNN implementation
│   │   ├── deepritz_solver.py # DeepRitz implementation
│   │   └── plotting.py     # Visualization utilities
│   └── matlab/         # MATLAB reference implementation
├── models/             # Model storage
│   ├── pinn_model_*    # Each trained PINN model with configs and history
│   ├── cnn_model_*     # Each trained CNN model with configs and history
│   └── deepritz_model_* # Each trained DeepRitz model with configs and history
├── assets/             # Output videos and frames
│   ├── fem/            # FEM simulation outputs
│   ├── pinn/           # PINN simulation outputs
│   ├── cnn/            # CNN simulation outputs
│   └── deepritz/       # DeepRitz simulation outputs
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

### Train Neural Network Models

#### PINN (Physics-Informed Neural Networks)
```bash
python -m src.python.main pinn-train [--model-name NAME]
```

#### CNN (Convolutional Neural Networks) 
```bash
python -m src.python.main cnn-train [--model-name NAME]
```

#### DeepRitz (Variational PINNs)
```bash
python -m src.python.main deepritz-train [--model-name NAME]
```

- `--model-name`: Optional name for the model directory (default: auto-generated timestamp)

Each training run creates a new directory in `models/` containing:
- `model_weights.pth`: The trained model weights
- `training_loss.png`: Plot of the loss curves during training
- `*_solver_script.py`: Copy of the solver script used for reproducibility

### Generate Predictions

#### PINN Predictions
```bash
python -m src.python.main pinn-predict --case [high|normal|low|all] [--model-name NAME]
```

#### CNN Predictions
```bash
python -m src.python.main cnn-predict --case [high|normal|low|all] [--model-name NAME]
```

#### DeepRitz Predictions
```bash
python -m src.python.main deepritz-predict --case [high|normal|low|all] [--model-name NAME]
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

2. Train models (choose one or more):
   ```
   # Train a PINN model
   python -m src.python.main pinn-train --model-name my_pinn_model
   
   # Train a CNN model
   python -m src.python.main cnn-train --model-name my_cnn_model
   
   # Train a DeepRitz model
   python -m src.python.main deepritz-train --model-name my_deepritz_model
   ```

3. Generate predictions with the trained models:
   ```
   python -m src.python.main pinn-predict --case all --model-name my_pinn_model
   python -m src.python.main cnn-predict --case all --model-name my_cnn_model
   python -m src.python.main deepritz-predict --case all --model-name my_deepritz_model
   ```

4. Compare the results in `assets/fem/`, `assets/pinn/`, `assets/cnn/`, and `assets/deepritz/`

## Methods Comparison

- **FEM**: Traditional finite element method, provides reference solution
- **PINN**: Physics-informed neural networks using strong form of the PDE
- **CNN**: Deep learning approach with U-Net architecture for spatiotemporal prediction
- **DeepRitz**: Variational physics-informed neural networks using weak form of the PDE (often more stable than traditional PINNs)

## Experiment Management

The system automatically tracks experiments by:
- Saving each model with a timestamp or custom name
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
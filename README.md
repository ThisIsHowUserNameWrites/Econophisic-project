# Ising Model for Econophysics Project

## Overview

This repository contains an implementation of the Ising model adapted for econophysics, focusing on the analysis of financial markets through statistical physics. The model helps understand correlation mechanisms between different financial assets and the emergence of collective behaviors in markets.

## Repository Structure

The repository is organized as follows:
```
Ising-Econophysics/ 
  ├── Ising MATLAB/ # Incomplete MATLAB implementation 
  ├── NoteBook/ # Jupyter notebook with complete implementation 
  └── Model_d_ISING_CODE.py # Main Python implementation file
```

## Components Description

### Ising MATLAB

This folder contains a partial implementation of the Ising model in MATLAB. While unfinished, it provides a foundation for those who prefer working with MATLAB. Key features:

- Basic structure for simulating spin networks
- Visualization capabilities for Hamiltonian components
- Framework for analyzing financial metrics

### NoteBook

The Jupyter notebook offers a complete, interactive implementation of the econophysics-adapted Ising model with:

- Detailed theoretical explanation of the model
- Full implementation of core functions
- Simulation for K=1 case (single financial asset)
- Simulation for K=2 case (two coupled assets)
- Analysis of coupling coefficient γ impact on correlations
- Visualization of spin networks and Hamiltonian layers
- Discussion of limitations and possible extensions

### Model_d_ISING_CODE.py

This Python file contains the complete implementation of the Ising model for econophysics, extracting the core functionality from the Jupyter notebook. It includes:

- All utility functions for simulation
- Visualization capabilities
- Hamiltonian decomposition analysis
- Main simulation workflow

## Installation and Dependencies
```
pip install numpy matplotlib
```


For the MATLAB version, you'll need MATLAB and import the installed file to a new project .

## Usage

### Python Implementation

1. Run the main Python file:


To run the Python implementation or the Jupyter notebook, you'll need:
```jupyter notebook```
3. Open the notebook and run cells sequentially

## Key Parameters

- `n`: Size of the network (n×n)
- `beta`: Temperature parameter (β = 1/T)
- `alpha`: Global state sensitivity parameter
- `gamma`: Coupling coefficient between assets
- `K`: Number of financial assets
- `T_steps`: Number of simulation steps

## References

- Mantegna, R. N., & Stanley, H. E. (1999). Introduction to econophysics: correlations and complexity in finance.
- Bouchaud, J. P., & Potters, M. (2003). Theory of financial risk and derivative pricing: from statistical physics to risk management.

## License

This project is provided for educational and research purposes.


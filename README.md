# Diagonal Schedule Syndrome Extraction - Surface Code Experiments

This repository contains implementations comparing standard fixed-bulk conventions vs diagonal schedules for surface code syndrome extraction in two scenarios:

## Files

### Core Implementation
- **`diagonal_plaquettes.py`**: Custom `DiagonalPlaquetteGenerator` that provides diagonal schedule plaquettes (schedule [6,4,3,5] for Z and [1,4,3,2] for X). Handles both memory experiments and X-junction spatial cubes.

### Comparison Scripts

#### 1. Memory Experiment Comparison (`complete_comparison.py`)
Compares standard vs diagonal schedule memory experiments:
```bash
# Basic comparison (k=1,2,3, default noise levels)
./venv/bin/python complete_comparison.py

# Custom k values
./venv/bin/python complete_comparison.py --k-values 2 3 4

# Custom shots and noise levels
./venv/bin/python complete_comparison.py --shots 100000 --noise-levels 0.001 0.002 0.003

# Skip certain analyses
./venv/bin/python complete_comparison.py --skip-distance --skip-logical-error
```

#### 2. X-Junction Comparison (`compare_x_junction.py`)
Compares standard vs diagonal schedule for spatial X-junctions:
```bash
./venv/bin/python compare_x_junction.py
```

Outputs include:
- Graph-like distance for both circuits
- Number of qubits, detectors, and observables
- Crumble URLs for visualization

## Key Features

- **Same Detector Count**: Both diagonal and standard circuits produce the same number of detectors, ensuring each ancilla connects to the same number of data qubits
- **Corner Handling**: Properly handles 3-body corner plaquettes when two adjacent arms are missing in spatial cubes
- **Boundary Plaquettes**: 2-body boundary plaquettes derived from diagonal bulk schedules with appropriate qubit omissions

## Requirements

- Python 3.13+
- Dependencies in `venv/` (activate with `source venv/bin/activate`)
- tqec library (installed in venv)


# Fluid-Flow-Simulation-Using-Physics-Informed-Neural-Networks

Develop a Physics-Informed Neural Network (PINN) for fluid flow simulation.

---

## Overview

- **Tooling & Environment:** MATLAB R2025b (Deep Learning Toolbox, Optimization Toolbox)
- **Status:** Completed

This repository contains the implementation of a Physics-Informed Neural Network (PINN) designed to resolve incompressible fluid flow profiles by directly coupling governing differential equations with deep learning architectures.

---

## Motivation

Predicting fluid behavior accurately remains central to design cycles in aerospace, automotive development, civil infrastructure, and environmental modeling. Whether analyzing boundary-layer separation across aerofoils to trim drag, sizing urban stormwater distribution networks, or tracking contaminant plumes through waterways, engineers rely heavily on continuous velocity and pressure field estimates.

Standard computational fluid dynamics (CFD) packages resolve these systems with high fidelity, but the underlying mesh generation and iterative numerical solvers impose heavy computational overhead—especially across parameterized sweeps and complex geometries. 

Physics-Informed Neural Networks bridge this gap by encoding the Navier–Stokes conservation equations directly into the network's loss objective. Instead of relying purely on dense labeled datasets, the solver enforces governing momentum and continuity residuals at chosen collocation points. Once convergence is reached, forward evaluation of the neural surrogate runs orders of magnitude faster than conventional discretizations, providing a viable reduced-order pathway for iterative design and fast inference.

## Phase 1: Dataset Selection and Physical Domain

### 1. Benchmark Problem Formulation
To establish a baseline for surrogate modeling and PINN convergence, unsteady incompressible laminar flow past an immersed cylinder (2D von Kármán vortex shedding) serves as the primary benchmark. 

The reference dataset is derived from high-resolution OpenFOAM simulations hosted in the open-source DeepCFD repository. Each spatial slice records spatial domain representations alongside resolved Navier–Stokes state variables across a regular discretized lattice.

- **Spatial Resolution:** $172 \times 79$ grid points
- **Input Channels (3):** Signed Distance Function (SDF) of the obstacle, normal distance to walls, and discrete boundary condition masks
- **Target Channels (3):** Streamwise velocity component ($U_x$), spanwise velocity component ($U_y$), and kinematic pressure ($p$)

---

### 2. Preprocessing & Data Ingestion Pipeline

The pipeline handles binary ingestion, format restructuring, numerical cleaning, and normalization. All ingestion logic is implemented in the executable Live Script:

`Data Collection and Preprocessing.mlx`

#### Execution Workflow
1. **Archive Ingestion:** Download `DeepCFD.zip` into the local working directory.
2. **Binary Conversion Bridge:** The raw pickle files (`dataX.pkl`, `dataY.pkl`) are parsed via a lightweight Python-to-MATLAB interface bridge, addressing NumPy array structures without serialization locks.
3. **Memory Layout Permutation:** Python array tensors stored in row-major layout (`[Batch, Channels, Height, Width]`) are mapped to native MATLAB deep learning spatial-batch layout (`[Height, Width, Channels, Batch]`).
4. **Data Verification:** Slices are scanned across all dimensions to discard instances containing unphysical singularities ($\pm\infty$, `NaN`).
5. **Feature Scaling:** Symmetrical min-max scaling maps inputs and targets to the range $[-1, 1]$:

$$x_{\text{norm}} = 2 \cdot \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}} - 1$$

This avoids early gradient saturation when passing states through anti-symmetric activation functions like $\tanh$.

6. **Partitioning:** The cleaned dataset is partitioned using stratified hold-out splits:
   - **Training Set:** 70%
   - **Validation Set:** 15%
   - **Test Set:** 15%

---

### 3. Execution Verification & Partition Output

Running `Data Collection and Preprocessing.mlx` parses the archive and exports standardized tensor structures to `DeepCFD_preprocessed.mat`.

#### Pipeline Console Output
```text
Extracting DeepCFD.zip...
Converting pickle binaries to MAT format via Python bridge...
Dataset loaded: 981 samples | Resolution: 172x79 | In-Channels: 3 | Out-Channels: 3

Pipeline executed successfully:
  Target Storage:       ./DeepCFD_preprocessed.mat
  Training partition:   [172 x 79 x 3 x 687]
  Validation partition: [172 x 79 x 3 x 147]
  Testing partition:    [172 x 79 x 3 x 147]

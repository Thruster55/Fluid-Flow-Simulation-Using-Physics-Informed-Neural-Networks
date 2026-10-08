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
```

## Phase 2: Model Architecture, Governing Physics, and Network Training

All routines for network architecture configuration, automatic differentiation of the Navier–Stokes residuals, boundary enforcement, and contour rendering are executed within:

`Model Development.mlx`

---

### 1. Network Topology & Collocation Setup

The surrogate model uses a fully connected Multilayer Perceptron (MLP) implemented in MATLAB via `dlnetwork`. The network approximates the continuous spatial mapping:

$$(x, y) \mapsto \big(u(x, y),\, v(x, y),\, p(x, y)\big)$$

where $x, y$ are spatial coordinates, $u, v$ represent Cartesian velocity components, and $p$ denotes kinematic pressure ($P/\rho$).

#### Architecture Configuration
| Layer Stage | Details | Layer Dimension | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | Coordinate tensor $(x, y)$ | $2 \times N$ | — |
| **Hidden Layers (1–4)** | 4 fully connected layers | $64$ neurons / layer | Hyperbolic Tangent ($\tanh$) |
| **Output** | Field state vector $(u, v, p)$ | $3 \times N$ | Linear |

#### Collocation Point Allocation
The 2D channel spans the bounded rectangular domain:

$$\Omega = [0.0,\, 2.0] \times [-0.5,\, 0.5]$$

- **Interior Domain ($N_f = 4000$):** Collocation points sampled uniformly across the channel interior to evaluate differential residuals.
- **Inlet Boundary ($x = 0.0$, $N_{\text{in}} = 200$):** Dirichlet inflow profile specified as $u = 1.0\ \text{m/s}$, $v = 0.0\ \text{m/s}$.
- **Solid Walls ($y = \pm 0.5$, $N_{\text{wall}} = 400$):** No-slip boundary condition enforced along the top and bottom plates: $u = 0.0\ \text{m/s}$, $v = 0.0\ \text{m/s}$.
- **Outlet Boundary ($x = 2.0$, $N_{\text{out}} = 200$):** Gauge reference pressure set to zero: $p = 0.0\ \text{m}^2/\text{s}^2$.

All collocation arrays are formatted as `dlarray` instances with Channel-Batch (`'CB'`) dimension labels for native execution.

---

### 2. Governing Equations & Loss Formulation

The loss objective is evaluated via `dlfeval` using reverse-mode automatic differentiation (`dlgradient`). First-order derivatives enable higher-order tracking (`'EnableHigherDerivatives', true`) to compute second-order diffusion terms directly through the computational graph.

#### Incompressible Navier–Stokes Residuals
For steady, laminar, 2D incompressible flow with kinematic viscosity $\nu = 0.01\ \text{m}^2/\text{s}$:

- **Continuity Residual:**
  $$e_{\text{cont}} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

- **$x$-Momentum Residual:**
  $$e_{\text{mom}, x} = \left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

- **$y$-Momentum Residual:**
  $$e_{\text{mom}, y} = \left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)$$

#### Multi-Term Loss Objective
$$\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{bc}} \mathcal{L}_{\text{bc}}$$

- **Residual Penalty:**
  $$\mathcal{L}_{\text{pde}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left( e_{\text{cont}}^2 + e_{\text{mom}, x}^2 + e_{\text{mom}, y}^2 \right)$$

- **Boundary Penalty:**
  $$\mathcal{L}_{\text{bc}} = \frac{1}{N_{\text{in}}} \sum \left( (u - U_{\text{in}})^2 + v^2 \right) + \frac{1}{N_{\text{wall}}} \sum \left( u^2 + v^2 \right) + \frac{1}{N_{\text{out}}} \sum p^2$$

- **Loss Weights:** $w_{\text{pde}} = 1.0$, $w_{\text{bc}} = 10.0$ (assigning higher initial priority to physical boundary compliance).

---

### 3. Optimization Setup & Training Log

The network parameters are optimized across 1000 epochs using the Adam algorithm (`adamupdate`) with an initial learning rate $\alpha = 10^{-3}$.

#### Execution Console Output
```text
Beginning PINN training with embedded Navier-Stokes loss...
Epoch:    1 | Total Loss: 8.8541e+00 | PDE Residual: 1.7565e-01 | BC Loss: 8.6785e-01
Epoch:  100 | Total Loss: 9.1248e-01 | PDE Residual: 2.3622e-01 | BC Loss: 6.7626e-02
Epoch:  200 | Total Loss: 5.0228e-01 | PDE Residual: 8.1792e-02 | BC Loss: 4.2049e-02
Epoch:  300 | Total Loss: 4.0719e-01 | PDE Residual: 6.5825e-02 | BC Loss: 3.4136e-02
Epoch:  400 | Total Loss: 3.8451e-01 | PDE Residual: 6.8610e-02 | BC Loss: 3.1590e-02
Epoch:  500 | Total Loss: 3.6726e-01 | PDE Residual: 6.9516e-02 | BC Loss: 2.9774e-02
Epoch:  600 | Total Loss: 3.5916e-01 | PDE Residual: 7.1721e-02 | BC Loss: 2.8744e-02
Epoch:  700 | Total Loss: 3.4992e-01 | PDE Residual: 7.2020e-02 | BC Loss: 2.7790e-02
Epoch:  800 | Total Loss: 3.4503e-01 | PDE Residual: 7.1935e-02 | BC Loss: 2.7309e-02
Epoch:  900 | Total Loss: 3.4285e-01 | PDE Residual: 7.3499e-02 | BC Loss: 2.6935e-02
Epoch: 1000 | Total Loss: 3.3690e-01 | PDE Residual: 6.9892e-02 | BC Loss: 2.6701e-02
```









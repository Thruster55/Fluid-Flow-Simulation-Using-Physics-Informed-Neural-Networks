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

## Phase 3: Hybrid PINN Training, Validation, and Ground-Truth Benchmarking

This stage couples data-driven supervision from high-fidelity OpenFOAM simulations with physical conservation laws to accelerate convergence and constrain the learned solution space. The execution pipeline, quantitative validation, error calculation, and model checkpointing are handled in:

`Training_Validation_Comparison.m`

---

### 1. Hybrid Loss Formulation & Collocation Grid

Rather than training purely as a black-box regression model or purely on unanchored physics residuals, the optimization objective balances empirical target fields with governing differential equations across the physical domain $\Omega = [0.0, 2.0] \times [-0.5, 0.5]\ \text{m}$.

The composite objective function is formulated as:

$$\mathcal{L}_{\text{total}} = w_{\text{data}} \mathcal{L}_{\text{data}} + w_{\text{pde}} \mathcal{L}_{\text{pde}}$$

where:
- $w_{\text{data}} = 10.0$ (supervised empirical anchor)
- $w_{\text{pde}} = 1.0$ (Navier–Stokes physical regularizer)
- $\nu = 0.01\ \text{m}^{2}/\text{s}$ (kinematic viscosity)

#### Empirical Data Loss ($\mathcal{L}_{\text{data}}$)
Measures the mean squared error (MSE) between network predictions $\hat{\mathbf{Y}} = [\hat{u}, \hat{v}, \hat{p}]^{T}$ and normalized OpenFOAM CFD ground-truth target values $\mathbf{Y} = [u^{\ast}, v^{\ast}, p^{\ast}]^{T}$:

$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left( (\hat{u}_{i} - u_{i}^{\ast})^{2} + (\hat{v}_{i} - v_{i}^{\ast})^{2} + (\hat{p}_{i} - p_{i}^{\ast})^{2} \right)$$

#### Physics Residual Loss ($\mathcal{L}_{\text{pde}}$)
Evaluates governing incompressible 2D Navier–Stokes residuals across domain collocation points via automatic differentiation (`dlgradient`):

$$\mathcal{L}_{\text{pde}} = \frac{1}{N} \sum_{i=1}^{N} \left( \vert{}e_{\text{cont}, i}\vert{}^{2} + \vert{}e_{\text{mom}, x, i}\vert{}^{2} + \vert{}e_{\text{mom}, y, i}\vert{}^{2} \right)$$

where:

$$e_{\text{cont}} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

$$e_{\text{mom}, x} = \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^{2} u}{\partial x^{2}} + \frac{\partial^{2} u}{\partial y^{2}} \right)$$

$$e_{\text{mom}, y} = \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^{2} v}{\partial x^{2}} + \frac{\partial^{2} v}{\partial y^{2}} \right)$$

---

### 2. Validation Metric: Relative $L_2$ Error Norm

Generalization is tracked across training epochs against an unseen hold-out validation slice $\mathbf{Y}_{\text{val}}$ using the relative $L_{2}$ error norm:

$$\text{Rel } L_{2} = \frac{\Vert{} \hat{\mathbf{Y}} - \mathbf{Y}_{\text{val}} \Vert{}_{2}}{\Vert{} \mathbf{Y}_{\text{val}} \Vert{}_{2}} = \frac{\sqrt{\sum_{i=1}^{N} \Vert{} \hat{\mathbf{y}}_{i} - \mathbf{y}_{\text{val}, i} \Vert{}^{2}}}{\sqrt{\sum_{i=1}^{N} \Vert{} \mathbf{y}_{\text{val}, i} \Vert{}^{2}}}$$

---

### 3. Optimization Setup & Training Log

The network is optimized over 1500 epochs using the Adam algorithm (`adamupdate`) with an initial learning rate $\alpha = 10^{-3}$. Metrics are recorded every 50 epochs.

#### Training Progression Log
```text
Beginning combined Data + Navier-Stokes PINN Training...
Epoch:    1 | Total Loss: 1.1426e+00 | Data Loss: 9.6686e-02 | PDE Residual: 1.7578e-01 | Val Rel L2: 5.3461e-01
Epoch:   50 | Total Loss: 3.7650e-02 | Data Loss: 3.5542e-03 | PDE Residual: 2.1082e-03 | Val Rel L2: 1.6639e-01
Epoch:  100 | Total Loss: 3.2665e-02 | Data Loss: 3.1724e-03 | PDE Residual: 9.4130e-04 | Val Rel L2: 1.6446e-01
Epoch:  200 | Total Loss: 3.1082e-02 | Data Loss: 3.0529e-03 | PDE Residual: 5.5287e-04 | Val Rel L2: 1.6363e-01
Epoch:  300 | Total Loss: 3.0562e-02 | Data Loss: 3.0088e-03 | PDE Residual: 4.7399e-04 | Val Rel L2: 1.6320e-01
Epoch:  400 | Total Loss: 3.0345e-02 | Data Loss: 2.9896e-03 | PDE Residual: 4.4854e-04 | Val Rel L2: 1.6287e-01
Epoch:  500 | Total Loss: 3.0221e-02 | Data Loss: 2.9784e-03 | PDE Residual: 4.3748e-04 | Val Rel L2: 1.6255e-01
Epoch:  600 | Total Loss: 3.0115e-02 | Data Loss: 2.9682e-03 | PDE Residual: 4.3201e-04 | Val Rel L2: 1.6218e-01
Epoch:  700 | Total Loss: 2.9992e-02 | Data Loss: 2.9556e-03 | PDE Residual: 4.3525e-04 | Val Rel L2: 1.6169e-01
Epoch:  800 | Total Loss: 2.9830e-02 | Data Loss: 2.9388e-03 | PDE Residual: 4.4201e-04 | Val Rel L2: 1.6116e-01
Epoch:  900 | Total Loss: 3.3509e-02 | Data Loss: 3.2336e-03 | PDE Residual: 1.1732e-03 | Val Rel L2: 1.6490e-01
Epoch: 1000 | Total Loss: 2.9287e-02 | Data Loss: 2.8746e-03 | PDE Residual: 5.4141e-04 | Val Rel L2: 1.5929e-01
Epoch: 1100 | Total Loss: 3.4414e-02 | Data Loss: 3.2629e-03 | PDE Residual: 1.7851e-03 | Val Rel L2: 1.7018e-01
Epoch: 1200 | Total Loss: 2.7567e-02 | Data Loss: 2.6785e-03 | PDE Residual: 7.8218e-04 | Val Rel L2: 1.5407e-01
Epoch: 1300 | Total Loss: 2.6984e-02 | Data Loss: 2.5296e-03 | PDE Residual: 1.6876e-03 | Val Rel L2: 1.4784e-01
Epoch: 1400 | Total Loss: 2.5513e-02 | Data Loss: 2.3622e-03 | PDE Residual: 1.8905e-03 | Val Rel L2: 1.4488e-01
Epoch: 1500 | Total Loss: 2.5087e-02 | Data Loss: 2.3234e-03 | PDE Residual: 1.8530e-03 | Val Rel L2: 1.4366e-01

Model successfully saved to: ./PINN_Trained_Model.mat







# Multi-Objective Optimisation for Smart Energy Grids

This repository implements a hybrid optimization framework combining **Bayesian Optimization (BO)** and **Mesh Adaptive Direct Search (MADS)** to address multi-objective prediction tasks in renewable energy markets. The primary focus is on optimizing trade-offs between **model accuracy** and **sparsity** in neural network architectures, particularly within the context of smart energy grids. Here we repeat the experiments from the paper: *Algorithm Switching for Multiobjective Predictions in Renewable Energy Markets*. (Implementation: https://scm.cms.hu-berlin.de/aswinkannan1987/lion)

## Overview

The project explores the application of **Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO)**, originally designed for high-dimensional optimization problems, to lower-dimensional scenarios such as the Unit Commitment problem in smart energy grids. By integrating SAASBO with MADS, the framework aims to efficiently navigate the search space, balancing exploration and exploitation, to identify optimal configurations that satisfy multiple objectives.

## Features

- **Multi-Objective Optimization**: Simultaneous optimization of conflicting objectives, specifically targeting model accuracy and sparsity.
- **SAASBO Integration**: Utilization of SAASBO to identify and focus on the most influential dimensions in the search space.
- **Hybrid Optimization Approach**: Combination of Bayesian Optimization for global search and MADS for local refinement.
- **Robust Error Handling**: Incorporation of mechanisms to handle numerical instabilities, such as NaNs and infinite gradients, ensuring the continuity of the optimization process.
- **Pareto Front Analysis**: Visualization and analysis of the trade-offs between objectives to aid in decision-making.

## Repository Structure

```
├── data/                       
├── utils/                    # Helper functions and utilities  
│   ├── Aquisition.py           
│   ├── Bayesian.py             
│   └── GaussianProcesses.py    
|   └-- Kernels.py
├── ResNet-demo.py           # Entry point for running experiments
|-- ResNet_model.py         # Original implementation from LION implementation
├── MOML_Report.pdf        
└── README.md               # Project overview and instructions
```

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Running Experiments

1. Prepare the dataset by placing it in the `data/` directory.
2. Configure the experiment parameters in `main.py`.
3. Execute the main script:

```bash
python ResNet-demo.py
```

4. Results, including Pareto front visualizations, will be saved in the `results/` directory.

## Results

The optimization framework successfully identifies configurations that balance accuracy and sparsity. However, observations indicate a degenerate Pareto front, suggesting potential areas for improvement:

- Re-evaluating the weighting scheme used in scalarizing objectives.
- Enhancing the surrogate model to better capture the trade-offs.
- Incorporating alternative multi-objective optimization strategies, such as Pareto-aware methods.


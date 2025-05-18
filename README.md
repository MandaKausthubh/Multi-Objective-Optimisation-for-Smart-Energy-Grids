# Multi-Objective Optimisation for Smart Energy Grids

This repository implements a hybrid optimization framework combining **Bayesian Optimization (BO)** and **Mesh Adaptive Direct Search (MADS)** to address multi-objective prediction tasks in renewable energy markets. The primary focus is on optimizing trade-offs between **model accuracy** and **sparsity** in neural network architectures, particularly within the context of smart energy grids.

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
├── data/                   # Dataset files for training and evaluation
├── models/                 # Neural network architectures and training scripts
├── optimization/           # Optimization algorithms and related utilities
│   ├── saasbo.py           # Implementation of SAASBO
│   ├── mads.py             # Implementation of MADS
│   └── hybrid_optimizer.py # Integration of BO and MADS
├── utils/                  # Helper functions and utilities
├── main.py                 # Entry point for running experiments
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install the required packages:

```bash
pip install -r requirements.txt
```

### Running Experiments

1. Prepare the dataset by placing it in the `data/` directory.
2. Configure the experiment parameters in `main.py`.
3. Execute the main script:

```bash
python main.py
```

4. Results, including Pareto front visualizations, will be saved in the `results/` directory.

## Results

The optimization framework successfully identifies configurations that balance accuracy and sparsity. However, observations indicate a degenerate Pareto front, suggesting potential areas for improvement:

- Re-evaluating the weighting scheme used in scalarizing objectives.
- Enhancing the surrogate model to better capture the trade-offs.
- Incorporating alternative multi-objective optimization strategies, such as Pareto-aware methods.


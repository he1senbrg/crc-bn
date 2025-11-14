# Colorectal Cancer Risk Prediction using Bayesian Networks

This project implements a complete pipeline for colorectal cancer risk mapping through Bayesian networks, including expert-seeded structure learning, parameter estimation with informative priors, temporal updating, and comprehensive evaluation.

## Project Structure

The codebase has been refactored into modular components for better maintainability and readability:

### Core Modules

- **`config.py`** - Configuration constants, variables, and expert constraints
- **`data_generator.py`** - Synthetic dataset generation for CRC risk factors
- **`structure_learning.py`** - Bayesian network structure learning with expert knowledge
- **`parameter_estimation.py`** - Parameter estimation with marginal priors
- **`temporal_updating.py`** - Temporal updating of model parameters across years
- **`prediction.py`** - Model prediction, evaluation, and performance metrics
- **`calibration.py`** - Model calibration using quantile binning
- **`visualization.py`** - Risk mapping and heatmap generation
- **`influence_analysis.py`** - Variable influence analysis and ranking
- **`main.py`** - Main pipeline orchestration

### Key Features

1. **Expert-seeded Structure Learning**: Uses hill-climb search with expert knowledge constraints
2. **Informative Dirichlet Priors**: Parameter estimation using marginal distributions
3. **Temporal Updating**: Progressive updating of model parameters across time periods
4. **Comprehensive Evaluation**: AUC, sensitivity, specificity, G-mean optimization
5. **Calibration Analysis**: Quantile binning for probability calibration assessment
6. **Risk Mapping**: Visualization of risk patterns across variables
7. **Influence Ranking**: Analysis of variable importance through perturbation

### Usage

Run the complete pipeline:
```bash
source .venv/bin/activate
python main.py
```

### Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pgmpy

### Pipeline Overview

1. **Data Generation**: Creates synthetic CRC dataset with known risk relationships
2. **Structure Learning**: Learns Bayesian network structure with expert constraints
3. **Parameter Estimation**: Fits parameters using maximum likelihood estimation
4. **Temporal Updating**: Updates model parameters across different years
5. **Model Evaluation**: Tests model performance on holdout data
6. **Calibration**: Assesses probability calibration quality
7. **Risk Analysis**: Creates risk maps and influence rankings

The refactored design separates concerns into focused modules while maintaining the complete functionality of the original monolithic script.

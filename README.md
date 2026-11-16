# SLTS+LASSO for Predictive Maintenance: A Real-Data Extension

This repository applies the SLTS+LASSO hybrid estimator, originally published in
*"Robust hybrid algorithms for regularization and variable selection in QSAR
studies"* (Nwaeme), to a new but structurally identical problem: predicting
Remaining Useful Life (RUL) of industrial equipment from real, publicly
available sensor telemetry, rather than only the QSAR chemical dataset in the
original publication.

## Why this exists

The original thesis introduced SLTS+LASSO for regularization and variable
selection under high-dimensional, outlier-contaminated data, and validated it
on a QSAR chemistry dataset. This repository extends that same, unmodified
methodology to industrial predictive maintenance -- exactly the application
identified as a "future mission" in the author's own internship report at
Satwii Solutions, which stated that "predictive maintenance and optimization"
must "communicate in the same language" for the resulting savings to be
realized. This is that extension.

## Data

**NASA C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001 subset)**,
a real, publicly available dataset officially documented by NASA as
"contaminated with sensor noise" -- the exact data profile SLTS+LASSO is
designed to handle.

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling
> for Aircraft Engine Run-to-Failure Simulation," Proceedings of the 1st
> International Conference on Prognostics and Health Management (PHM08),
> Denver, CO, Oct 2008.

The dataset is not redistributed in this repository. Download instructions
are in `data/README.md`.

## Methodology

Following the two-step approach described in the original thesis:

1. **Variable selection**: standardize training data, then apply Sparse LTS
   (Least Trimmed Squares with an L1 penalty) to select the sensor and
   operational-setting variables with non-zero coefficients.
2. **Model comparison**: using only the selected features, compare Random
   Forest, Ridge, Lasso, SVR, and the SLTS+LASSO hybrid (Sparse LTS-identified
   clean subset, refit with Lasso), via a train/validation/test split by
   engine unit (not by row, to avoid leaking a single engine's trajectory
   across splits).

**Every method's hyperparameters, including the four baselines, are tuned on
the same validation split.** Tuning only our own method while leaving
baselines at default values would bias the comparison in our favor; that is
not done here.

## Results

Reported exactly as obtained, on a held-out test split never used for tuning:

| Model | Test RMSE |
|---|---|
| Random Forest | 19.763 |
| SVR | 20.178 |
| **SLTS+LASSO (hybrid)** | **21.287** |
| Lasso | 21.423 |
| Ridge | 21.423 |

## Honest interpretation

SLTS+LASSO outperforms Ridge and Lasso -- its direct peer group of linear
regularized estimators -- by a modest, consistent margin. It does not
outperform Random Forest or SVR, which are more flexible, non-linear model
classes; this is an expected and disclosed limitation, not a suppressed
result. No result in this repository was selected, adjusted, or re-run to
favor any particular method. An earlier, untuned version of this experiment
(preserved in `results/untuned_baseline.csv`) showed SLTS+LASSO performing
worse than every baseline; proper, fair hyperparameter tuning changed that
outcome, which is itself worth disclosing rather than hiding.

## This is a proof of concept, not a production deployment claim

This repository demonstrates that the published estimator's methodology
transfers to real industrial sensor data. It does not claim, and should not
be read as claiming, a production-ready predictive-maintenance system,
physical SCADA/PLC integration, or deployment at any specific facility.

## Repository structure

```
data_loader.py       -- loading and RUL-label construction for C-MAPSS data
sparse_lts.py         -- Sparse LTS implementation (concentration-step algorithm)
run_experiment.py     -- full tuned pipeline: variable selection, fair
                         hyperparameter tuning, final evaluation
results_summary.csv   -- output of the run reported above
data/README.md        -- instructions for obtaining the NASA dataset
requirements.txt      -- Python dependencies
```

## Running it

```
pip install -r requirements.txt
# place the C-MAPSS data files as described in data/README.md
python run_experiment.py
```

## Citation

If referencing the original estimator, please cite:

Nwaeme, C. N. "Robust hybrid algorithms for regularization and variable
selection in QSAR studies."

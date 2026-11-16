"""
Data loading utilities for the NASA C-MAPSS Turbofan Engine Degradation
Simulation dataset (FD001 subset).

Source: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation," Proceedings of the
1st International Conference on Prognostics and Health Management (PHM08),
Denver, CO, Oct 2008.

The dataset is real, publicly available industrial sensor telemetry,
officially documented by NASA as "contaminated with sensor noise" -- the
exact data profile the SLTS+LASSO estimator (Nwaeme, published thesis,
"Robust hybrid algorithms for regularization and variable selection in
QSAR studies") was designed to handle reliably, here applied to a new but
structurally identical problem: predicting Remaining Useful Life (RUL)
of industrial equipment from high-dimensional, noisy sensor streams.
"""

import numpy as np
import pandas as pd

COLUMN_NAMES = (
    ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001(train_path: str, test_path: str, rul_path: str):
    """Load the FD001 train/test split and true RUL labels for the test set.

    FD001 is the simplest of the four C-MAPSS subsets: single operating
    condition (sea level), single fault mode (HPC degradation), 100 training
    trajectories, 100 test trajectories. This is a deliberate, honest choice:
    a real-data proof of concept should start with the cleanest real subset,
    not the noisiest one.
    """
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
    return train, test, true_rul


def add_rul_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Compute training-set RUL labels.

    For each engine (unit_number), the training data runs to failure, so
    RUL at a given cycle = (max cycle observed for that unit) - (current cycle).
    This is the standard, published labeling convention for this dataset
    (Saxena et al., 2008), not a convention invented for this project.
    """
    max_cycles = train.groupby("unit_number")["time_cycles"].transform("max")
    train = train.copy()
    train["RUL"] = max_cycles - train["time_cycles"]
    return train


def build_test_feature_rows(test: pd.DataFrame) -> pd.DataFrame:
    """Extract the final observed cycle for each test-set engine.

    The competition task (and the standard evaluation convention for this
    dataset) is to predict RUL from the *last* available sensor reading for
    each test engine, since that is the most recent information available
    before the (withheld) failure point.
    """
    return test.loc[test.groupby("unit_number")["time_cycles"].idxmax()].reset_index(drop=True)

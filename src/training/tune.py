"""
Optuna hyperparameter search across backbone architectures.

This script is responsible for finding a good temporal backbone
configuration before the final training pipeline is run.

The backbone is the neural network that processes a sequence of
engine sensor measurements and converts that sequence into a learned
feature representation.

The architectures currently considered are:

    - LSTM
    - GRU
    - TCN
    - Transformer

Optuna searches over both:

    1. Which architecture to use
    2. The hyperparameters for that architecture

The best configuration is written to model_registry.yaml.

The later train.py script reads that configuration and performs the
full training process, including NGBoost and uncertainty calibration.

Usage
-----
    python src/training/tune.py --dataset FD001 --n_trials 50

After the study completes, the best trial's config is written into
config/model_registry.yaml under the dataset key. train.py then reads
that config to do the full training run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

# Project paths
# __file__ points to:
#     src/training/tune.py
# parents[2] therefore takes us back to the root of the project.
# This allows the script to construct paths such as:
#     data/processed/
#     models/
#     studies/
#     config/
# without depending on the directory from which the user launches
# the Python command.
_ROOT = Path(__file__).resolve().parents[2]

_TRAINING = _ROOT / "src" / "training"

# Make sure Python can find the shared training utilities.
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))


from shared import (
    DEVICE,
    build_backbone,
    load_registry,
    make_loader,
    save_registry,
    split_by_engine,
    train_backbone,
)

# Model Registry
def update_registry(
    registry_path: Path,
    dataset: str,
    backbone_name: str,
    best_rmse: float,
    backbone_cfg: dict,
    input_dim: int,
    seq_length: int,
    study_path: str,
) -> None:
    """
    Update the model registry with the current best backbone.

    The registry acts as a record of which model configuration has
    been selected as the current "champion" for a dataset.

    This is useful because the Optuna study may contain many trials,
    but the rest of the training pipeline only needs to know which
    configuration won.

    Parameters
    ----------
    registry_path:
        Location of model_registry.yaml.

    dataset:
        Dataset being optimized, e.g. "FD001".

    backbone_name:
        Name of the selected architecture.

    best_rmse:
        Validation RMSE achieved by the best trial.

    backbone_cfg:
        Architecture-specific hyperparameters.

    input_dim:
        Number of input features at each time step.

    seq_length:
        Number of historical time steps supplied to the backbone.

    study_path:
        Location of the Optuna study database.
    """

    # Load the existing registry rather than replacing it.
    # This is important because the registry contains entries for
    # multiple datasets. We only want to modify the selected dataset.
    registry = load_registry(registry_path)

    # Make sure the dataset has a dictionary entry.
    registry.setdefault(dataset, {})

    # Store the selected configuration as the dataset's champion.
    # "meta_model" is left as None because this script only chooses
    # the backbone. train.py later trains and records the NGBoost
    # meta-model.
    artifact_dir = str(_ROOT / "models" / dataset)

    registry[dataset]["champion"] = {
        "backbone": backbone_name,

        # train.py will fill this in after the NGBoost stage.
        "meta_model": None,

        # Combine architecture-specific settings with the two pieces
        #     input_dim
        #     seq_length
        "backbone_config": {
            **backbone_cfg,
            "input_dim": input_dim,
            "seq_length": seq_length,
        },

        # At this point only the backbone has been evaluated.
        # The remaining metrics are deliberately left empty until
        # train.py evaluates the complete backbone + NGBoost system.
        "metrics": {
            "val_rmse": float(best_rmse),
            "val_mae": None,
            "val_nll": None,
            "nasa_score": None,
        },

        # Locations where the trained artifacts will be stored.
        "artifacts": {
            "backbone": str(Path(artifact_dir) / "backbone.pt"),
            "meta_ngboost": str(Path(artifact_dir) / "stacked_ngboost.pkl"),
        },

        "trained_at": str(date.today()),

        # Keep the Optuna study location so that the experiment can
        # later be inspected or resumed.
        "optuna_study": study_path,
    }

    # Write the modified registry back to disk.
    save_registry(registry_path, registry)

    print(
        f" Registry updated: {registry_path}  "
        f"[{dataset} champion: {backbone_name} "
        f"RMSE={best_rmse:.4f}]"
    )

# Optuna Objective
def objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_cfg: dict,
    dataset: str,
) -> float:
    """
    Define one Optuna trial.

    Optuna calls this function repeatedly.

    Each call:

        1. Selects a backbone architecture.
        2. Selects its hyperparameters.
        3. Creates sequence DataLoaders.
        4. Builds the corresponding neural network.
        5. Trains the network.
        6. Returns validation RMSE.

    Optuna then uses the returned RMSE to decide which
    configurations are promising.

    The objective is deliberately based on validation data.

    The test set is not used here because it should remain an
    untouched final evaluation set.
    """

    # Choose the backbone architecture
    # Optuna treats the architecture itself as a hyperparameter.
    # This means we're not assuming ahead of time that an LSTM is best.
    # The experiment can compare several different ways of modelling
    # temporal relationships.
    backbone_name = trial.suggest_categorical("backbone", ["lstm", "gru", "tcn", "transformer"])

    # Shared hyperparameters
    # These parameters have a meaning across multiple architectures.
    # hidden_dim:
    #     Size of the learned representation.
    # dropout:
    #     Regularization intended to reduce overfitting.
    # lr:
    #     Learning rate used during neural-network training.
    # seq_length:
    #     Number of historical time steps supplied to the model.
    # This last parameter is particularly important for predictive
    # maintenance because it determines how much history the model
    # can see when estimating engine health.
    hidden_dim = trial.suggest_int( "hidden_dim",64, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Search over multiple sequence lengths rather than hardcoding
    # one historical window.
    # For example:
    #     20 cycles
    #     30 cycles
    #     40 cycles
    #     50 cycles
    # This allows the experiment to determine whether shorter or
    # longer temporal context is more useful.
    seq_length = trial.suggest_int("seq_length", 20, 50, step=10)

    # Architecture-specific hyperparameters
    if backbone_name in ("lstm", "gru"):

        # Number of recurrent layers.
        # More layers allow a deeper temporal representation but also
        # increase model complexity and the possibility of overfitting.
        num_layers = trial.suggest_int("num_layers", 1, 3)

        # A bidirectional recurrent network can process information
        # in both directions through the provided sequence.
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        backbone_cfg = dict(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    elif backbone_name == "tcn":
        # Number of temporal convolution levels.
        # Increasing the number of levels allows the TCN to capture
        # temporal patterns across increasingly large receptive fields.
        n_levels = trial.suggest_int("tcn_levels",3,6)

        # Kernel size determines how many neighboring time steps
        # a convolution considers at each layer.
        kernel_size = trial.suggest_int("kernel_size", 2,5)

        # TCN uses a list describing the number of channels at each level.
        backbone_cfg = dict(
            num_channels=[hidden_dim] * n_levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    elif backbone_name == "transformer":

        # Number of attention heads.
        # Each head learns a different attention representation of
        # the sequence.
        nhead = trial.suggest_categorical("nhead",[2, 4, 8])

        # Number of Transformer encoder layers.
        num_layers = trial.suggest_int("tf_layers", 1, 4)

        # Size of the feed-forward component inside each Transformer layer.
        dim_ff = trial.suggest_int("dim_feedforward", 128, 512, step=128,)

        # Transformer dimensionality constraint
        # PyTorch's multi-head attention divides d_model across the
        # attention heads.
        # Therefore:
        #     hidden_dim % nhead == 0
        # must be true.
        # Rather than allowing an invalid configuration to continue,
        # the trial is discarded.
        if hidden_dim % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        backbone_cfg = dict(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
        )

    # Build DataLoaders
    # Sequence length is part of the Optuna search, so the DataLoaders
    # have to be rebuilt for every trial.
    # Training data is shuffled.
    # Validation data is NOT shuffled.
    # Keeping validation order stable is important when later outputs
    # need to correspond to the original target ordering.
    train_loader = make_loader(
        train_df,
        seq_length=seq_length,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )

    val_loader = make_loader(
        val_df,
        seq_length=seq_length,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )

    # If the selected sequence length is too large for the available
    # data, no sequences may be created.
    # There is nothing useful to train in that situation, so the
    # Optuna trial is discarded.
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise optuna.exceptions.TrialPruned()

    # Determine input dimensionality
    # Each sequence has approximately this structure:
    #     (batch, sequence_length, input_features)
    # The final dimension is the number of features available at each
    # time step.
    # We obtain it from an actual batch instead of hardcoding it.
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]

    # Build the selected neural-network architecture
    # build_backbone() acts as a factory:
    #     "lstm"        -> LSTM backbone
    #     "gru"         -> GRU backbone
    #     "tcn"         -> TCN backbone
    #     "transformer" -> Transformer backbone
    # A configuration that fails to construct is treated as an invalid
    # Optuna trial rather than terminating the entire search.
    try:
        model = build_backbone(
            backbone_name,
            input_dim,
            backbone_cfg,
        )
    except Exception:
        raise optuna.exceptions.TrialPruned()

    # Train the backbone
    # train_backbone() handles the actual neural-network training.
    # The function returns:
    #     model
    #     best validation RMSE
    # Optuna only needs the validation RMSE because that is the objective
    # it is trying to minimize.
    _, best_val_rmse = train_backbone(
        model,
        train_loader,
        val_loader,
        cfg={
            **train_cfg,
            "lr": lr,
        },
        trial=trial,
        verbose=False,
    )

    # Track the best trial
    # Optuna already knows which trial is best, but this additionally
    # stores the current global best in the study's user attributes.
    # This is used below to save the actual backbone whenever a new
    # best configuration is found.
    global_best = trial.study.user_attrs.get(
        "global_best_rmse",
        float("inf"),
    )

    if best_val_rmse < global_best:

        trial.study.set_user_attr(
            "global_best_rmse",
            best_val_rmse,
        )

        # Save the current best model.
        # This means the best backbone weights are available even before
        # the complete training pipeline runs.
        model_dir = _ROOT / "models" / dataset
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            model_dir / "backbone.pt",
        )

        # Record the winning configuration in the central registry.
        # train.py will later read this information when performing the
        # full backbone + NGBoost training.
        update_registry(
            registry_path=_ROOT / "config" / "model_registry.yaml",
            dataset=dataset,
            backbone_name=backbone_name,
            best_rmse=best_val_rmse,
            backbone_cfg=backbone_cfg,
            input_dim=input_dim,
            seq_length=seq_length,
            study_path=str(
                _ROOT / "studies" / f"{dataset}_v1.db"
            ),
        )

    # The value returned here is what Optuna minimizes.
    return best_val_rmse


# Command-Line Arguments

def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the Optuna search.

    Keeping these values configurable means the same script can be used
    for quick experiments as well as larger searches.
    """

    parser = argparse.ArgumentParser(
        description="Backbone architecture search via Optuna."
    )

    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default="FD001",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max epochs per trial.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience per trial.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    return parser.parse_args()


# Main Program

if __name__ == "__main__":

    # Read command-line configuration.
    args = parse_args()

    # The architecture search operates on the processed feature file.
    feature_path = (
        _ROOT
        / "data"
        / "processed"
        / args.dataset
        / "train_features.csv"
    )

    # If preprocessing/feature engineering has not been run yet,
    # there is nothing for the backbone search to train on.
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            f"Run preprocessing and feature engineering first:\n"
            f"  python src/data/preprocessing.py "
            f"--dataset {args.dataset}\n"
            f"  (then run the feature engineering notebook)"
        )

    # Load the processed dataset.
    full_df = pd.read_csv(feature_path)

    # Split by ENGINE rather than by individual rows.
    # This is important for predictive maintenance because rows from
    # the same engine are temporally related.
    # Splitting individual rows could allow information from the same
    # physical engine to appear in both train and validation sets.
    train_df, val_df = split_by_engine(full_df)

    print(f"Dataset: {args.dataset}")
    print(f"Device: {DEVICE}")

    print(
        f"Train: {train_df['unit_number'].nunique()} engines  "
        f"({len(train_df):,} rows)"
    )

    print(
        f"Val : {val_df['unit_number'].nunique()} engines  "
        f"({len(val_df):,} rows)"
    )

    print(f"Trials: {args.n_trials}\n")

    # Configuration shared across all Optuna trials.
    # The individual trial's learning rate is added inside objective().
    train_cfg = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
    }

    # Optuna study storage
    # The study is stored in SQLite so that:
    #     - experiments survive program interruption
    #     - previous trials can be inspected
    #     - the study can be resumed
    # This is particularly useful for longer searches.
    studies_dir = _ROOT / "studies"
    studies_dir.mkdir(exist_ok=True)

    study_db = studies_dir / f"{args.dataset}_v1.db"

    study = optuna.create_study(
        direction="minimize",

        # MedianPruner can stop poorly performing trials early.
        # This prevents spending the full epoch budget on configurations
        # that are already performing significantly worse than previous
        # trials.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),

        storage=f"sqlite:///{study_db}",

        study_name=f"{args.dataset}_backbone_search",

        # If the study already exists, continue it instead of creating
        # a completely new experiment.
        load_if_exists=True,
    )

    # Restore the global-best attribute if it already exists.
    # If this is a new study, initialize it to infinity so that the
    # first successful trial automatically becomes the best.
    study.set_user_attr(
        "global_best_rmse",
        study.user_attrs.get(
            "global_best_rmse",
            float("inf"),
        ),
    )

    # Run the requested number of trials.
    #
    # Each trial calls objective(), which chooses an architecture,
    # trains it, evaluates validation RMSE, and returns that value.
    study.optimize(
        lambda trial: objective(
            trial,
            train_df,
            val_df,
            train_cfg,
            args.dataset,
        ),
        n_trials=args.n_trials,
    )

    # Report the final result

    print(f"Optimization complete - {args.dataset}")
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best RMSE  : {study.best_value:.4f}")

    print("Best params:")

    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
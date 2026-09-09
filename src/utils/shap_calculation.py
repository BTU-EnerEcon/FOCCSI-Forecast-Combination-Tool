"""Reduced SHAP computations used by the FOCCSI rolling-window pipeline."""

from itertools import combinations
from math import comb
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone


def _predict_reduced_game(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    x_row: pd.Series,
    coalition: List[str],
    model_cache: Dict,
) -> float:
    """Predict one row for a coalition by refitting/caching coalition-specific models."""
    if not coalition:
        return np.nan

    coalition_key = tuple(coalition)
    coalition_model = model_cache.get(coalition_key)
    if coalition_model is None:
        coalition_model = clone(model)
        coalition_model.fit(X_train[coalition], y_train)
        model_cache[coalition_key] = coalition_model

    x_input = pd.DataFrame([x_row[coalition]], columns=coalition)
    return float(coalition_model.predict(x_input)[0])


def _compute_sv_pb_rg_zci(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute sv_pb_rg_zci feature attributions for each row in X_test."""
    features = list(X_train.columns)
    n_features = len(features)
    shap_values = {feature: [] for feature in features}

    if n_features <= 1:
        return pd.DataFrame(shap_values, index=X_test.index)

    model_cache = {}
    for row_idx in range(len(X_test)):
        x_row = X_test.iloc[row_idx]
        y_true = float(y_test.iloc[row_idx])

        for feature in features:
            other_features = [feat for feat in features if feat != feature]
            phi = 0.0

            # sv_pb_rg_zci ignores the empty coalition and uses LASMO-style normalization.
            for coalition_size in range(1, len(other_features) + 1):
                coalition_weight = 1.0 / ((n_features - 1) * comb(n_features - 1, coalition_size))
                for coalition in combinations(other_features, coalition_size):
                    coalition = list(coalition)
                    pred_without = _predict_reduced_game(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        x_row=x_row,
                        coalition=coalition,
                        model_cache=model_cache,
                    )
                    pred_with = _predict_reduced_game(
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        x_row=x_row,
                        coalition=coalition + [feature],
                        model_cache=model_cache,
                    )

                    value_without = -((y_true - pred_without) ** 2)
                    value_with = -((y_true - pred_with) ** 2)
                    phi += coalition_weight * (value_with - value_without)

            shap_values[feature].append(phi)

    return pd.DataFrame(shap_values, index=X_test.index)


def compute_window_shap_results(
    model,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    y_pred,
    window_index: int,
    test_date,
    target_name: str,
    shap_cfg: Dict,
) -> pd.DataFrame:
    """Build a long-format SHAP table for one rolling window when enabled."""
    if not bool((shap_cfg or {}).get("sv_pb_rg_zci", False)):
        return pd.DataFrame()

    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] != 1:
            raise ValueError("sv_pb_rg_zci requires a single target column in y_train.")
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError("sv_pb_rg_zci requires a single target column in y_test.")
        y_test = y_test.iloc[:, 0]

    sv_pb_rg_zci_df = _compute_sv_pb_rg_zci(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Export as long-format rows to simplify downstream storage/analysis.
    records = []
    for timestamp in X_test.index:
        for feature in X_test.columns:
            records.append(
                {
                    "window_index": window_index,
                    "test_date": pd.Timestamp(test_date),
                    "target_name": target_name,
                    "time": timestamp,
                    "feature": feature,
                    "sv_pb_rg_zci": float(sv_pb_rg_zci_df.loc[timestamp, feature]),
                }
            )

    return pd.DataFrame(records)
"""Rolling-window model training and evaluation for FOCCSI."""

import numpy as np
import pandas as pd
import time
from datetime import timedelta
from sklearn.linear_model import ElasticNetCV
# PSO
from pyMetaheuristic.algorithm import particle_swarm_optimization
from src.utils.data_processing_utils import (
    apply_benchmark_threshold_filter,
    contains_invalid_values,
    get_latest_available_target_date_df,
    merge_price_data,
    preprocess_forecast_column,
    safe_rmse,
)
from src.utils.shap_calculation import compute_window_shap_results


class RunAborted(Exception):
    """Raised when a running pipeline is cancelled by the user."""



def _run_pso(X_train, y_train, X_test, n_features, objective_function="energy_error", price_spread=None, pso_params=None):
    """Fit PSO-based ensemble weights and predict the current test horizon."""
    X_train_vals = X_train.values
    y_train_vals = y_train.values
    X_test_vals = X_test.values

    def pso_objective(weights):
        """PSO objective function for pso model"""
        weights = np.clip(weights, 0, 1)
        total = np.sum(weights)
        if total == 0:
            weights = np.full_like(weights, 1 / len(weights))  # fall back to equal weights
        else:
            weights /= total
        preds = np.dot(X_train_vals, weights)
        error = y_train_vals - preds

        if objective_function == "balancing_cost_error":
            if price_spread is None:
                raise ValueError(
                    "Price spread is required for PSO objective_function='balancing_cost_error'."
                )
            spread_vals = np.asarray(price_spread)
            if spread_vals.shape[0] != error.shape[0]:
                raise ValueError("Price spread length does not match PSO training data length.")
            return np.mean((error * spread_vals) ** 2)

        return np.mean(error ** 2)

    pso_params = pso_params or {}
    swarm_size = pso_params.get("swarm_size", 30)
    start_init = np.full((swarm_size, n_features), 1 / n_features)
    pso_solver_params = {
        "target_function": pso_objective,
        "swarm_size": swarm_size,
        "min_values": np.zeros(n_features),
        "max_values": np.ones(n_features),
        "iterations": pso_params.get("iterations", 50),
        "decay": pso_params.get("decay", 0),
        "w": pso_params.get("w", 0.9),
        "c1": pso_params.get("c1", 2),
        "c2": pso_params.get("c2", 2),
        "verbose": False,
        "start_init": start_init,
        "target_value": None,
    }

    best_weights = particle_swarm_optimization(**pso_solver_params)[:-1]
    best_weights = np.clip(best_weights, 0, 1)
    total = np.sum(best_weights)
    if total == 0:
        best_weights = np.full_like(best_weights, 1 / len(best_weights))  # fall back to equal weights
    else:
        best_weights /= total
    y_pred = np.dot(X_test_vals, best_weights)
    return y_pred, best_weights



def rolling_window_of_timeseries(df, time_col, forecast_cols, benchmark_cols, secondary_benchmark_cols,
            train_days, test_days=1, model_config=None, progress_callback=None, status_callback=None,
            cancel_event=None, first_test_date=None, last_test_date=None):
    """Run rolling training/testing windows and return forecasts, RMSE rows, and SHAP rows."""
    start = time.time()

    # Normalize to a datetime index for reliable .loc time slicing.
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in dataframe columns.")

    df_work = df.copy()
    df_work[time_col] = pd.to_datetime(df_work[time_col], errors="coerce")
    df_work = df_work.dropna(subset=[time_col]).sort_values(time_col)
    if df_work.empty:
        raise ValueError("No valid timestamps found in time column after conversion.")


    df_work = df_work.set_index(time_col, drop=False).sort_index()

    # Read model and preprocessing configuration.
    model_cfg = (model_config or {}).get("models", {})
    run_delnet = model_cfg.get("delnet", False)
    run_pso = model_cfg.get("pso", False)
    run_pso_objective = model_cfg.get("pso_objective", "rmse")
    elasticnet_params_cfg = model_cfg.get("elasticnet_params", {})
    delnet_l1_ratio = elasticnet_params_cfg.get("l1_ratio", model_cfg.get("l1_ratio", 0.5))
    delnet_cv = elasticnet_params_cfg.get("cv", model_cfg.get("cv", 10))
    delnet_eps = elasticnet_params_cfg.get("eps", 0.001)
    delnet_n_alphas = elasticnet_params_cfg.get("n_alphas", 100)
    delnet_fit_intercept = elasticnet_params_cfg.get("fit_intercept", True)
    delnet_precompute = elasticnet_params_cfg.get("precompute", "auto")
    delnet_max_iter = elasticnet_params_cfg.get("max_iter", 1000)
    delnet_tol = elasticnet_params_cfg.get("tol", 0.0001)
    delnet_copy_X = elasticnet_params_cfg.get("copy_X", True)
    delnet_verbose = elasticnet_params_cfg.get("verbose", 0)
    delnet_n_jobs = elasticnet_params_cfg.get("n_jobs", None)
    delnet_positive = elasticnet_params_cfg.get("positive", False)
    delnet_random_state = elasticnet_params_cfg.get("random_state", None)
    delnet_selection = elasticnet_params_cfg.get("selection", "cyclic")
    pso_params_cfg = model_cfg.get("pso_params", {})
    shap_cfg = model_cfg.get("shap_config", {})
    shap_enabled = model_cfg.get("shap_enabled", False)

    preprocessing_cfg = (model_config or {}).get("preprocessing", {})
    run_prefiltering = bool(preprocessing_cfg.get("filter_forecast_data", True))
    run_benchmark_filtering = bool(preprocessing_cfg.get("filter_benchmark_data", False))
    benchmark_threshold = preprocessing_cfg.get("benchmark_threshold", None)

    default_criteria = {
        "IV": 5, "NN": 5, "RN": 10, "RZ": 68, "RV": 10,
        "HIV": 100, "HNN": 96, "HRN": 960, "HRZ": 960, "HRV": 960,
    }
    criteria_cfg = preprocessing_cfg.get("forecast_thresholds", {})
    horizon_criteria_cfg = criteria_cfg.get("horizon_window", {})
    training_criteria_cfg = criteria_cfg.get("training_window", {})
    criteria_dict = {
        key: (
            training_criteria_cfg.get(key, criteria_cfg.get(key, default_value))
            if key.startswith("H")
            else horizon_criteria_cfg.get(key, criteria_cfg.get(key, default_value))
        )
        for key, default_value in default_criteria.items()
    }

    input_data_cfg = (model_config or {}).get("input_data", {})
    secondary_benchmark_df = input_data_cfg.get("secondary_benchmark_df")
    secondary_benchmark_cutoff_df = input_data_cfg.get("secondary_benchmark_cutoff_df")
    use_secondary_benchmark = secondary_benchmark_df is not None
    day_ahead_df = input_data_cfg.get("day_ahead_df")
    intraday_df = input_data_cfg.get("intraday_df")
    price_spread_data_available = day_ahead_df is not None and intraday_df is not None
    price_spread_df = merge_price_data(df_work, day_ahead_df, intraday_df) if price_spread_data_available else None
    primary_target_col = benchmark_cols[0] if benchmark_cols else input_data_cfg.get("target", "EUZ")
    secondary_target_col = secondary_benchmark_cols[0] if secondary_benchmark_cols else "HR"

    # Build an installed-capacity lookup series for normalizing predictions/RMSE by timestamp.
    normalize_output = bool((model_config or {}).get("postprocessing", {}).get("normalize_output", False))
    installed_capacity_df = input_data_cfg.get("installed_capacity_df")
    capacity_series = None
    if normalize_output and installed_capacity_df is not None and not installed_capacity_df.empty:
        capacity_time_col = "time" if "time" in installed_capacity_df.columns else installed_capacity_df.columns[0]
        capacity_value_cols = [col for col in installed_capacity_df.columns if col != capacity_time_col]
        if capacity_value_cols:
            cap_df = installed_capacity_df.copy()
            cap_df[capacity_time_col] = pd.to_datetime(cap_df[capacity_time_col], errors="coerce")
            cap_df = cap_df.dropna(subset=[capacity_time_col]).drop_duplicates(subset=[capacity_time_col], keep="last")
            capacity_series = cap_df.set_index(capacity_time_col)[capacity_value_cols[0]].sort_index()

    if run_benchmark_filtering:
        df_work = apply_benchmark_threshold_filter(df_work, primary_target_col, benchmark_threshold)

    if use_secondary_benchmark and (secondary_benchmark_cutoff_df is None or secondary_benchmark_cutoff_df.empty):
        raise ValueError(
            "secondary_benchmark_cutoff_df is required when a secondary benchmark dataset is provided."
        )
    
    # Configure PSO objective variant.
    pso_variant = None
    if run_pso:
        if run_pso_objective == "rmse":
            pso_variant = ("PSO", "energy_error")
        elif run_pso_objective == "balancing_price_error":
            pso_variant = ("PSO_balancing_price_objective_function", "balancing_cost_error")
        else:
            raise ValueError(f"Unknown PSO objective function: {run_pso_objective}")

    cutoff_df = secondary_benchmark_cutoff_df

    # Compute the rolling evaluation range.
    if not first_test_date:
        first_test_date = df_work.index.min() + timedelta(days=train_days)
    if not last_test_date:
        last_test_date = df_work.index.max() - timedelta(days=test_days)
    total_windows = int(((last_test_date - first_test_date).total_seconds() // 86400) + 1)
    total_windows = max(total_windows, 0)
    
    i = 0
    results = []
    rmse_records = []
    shap_results = []
    test_date = first_test_date
    processed_windows = 0

    def _emit_progress(current_test_date):
        """Publish inner-loop progress/status to outer UI callbacks."""
        if total_windows <= 0:
            local_progress = 1.0
        else:
            local_progress = min(1.0, max(0.0, processed_windows / total_windows))

        if progress_callback is not None:
            progress_callback(local_progress)

        if status_callback is not None:
            status_callback(
                f"Rolling window {processed_windows}/{total_windows} | Test: {current_test_date.date()}"
            )

    def _advance_window(current_test_date):
        nonlocal processed_windows, test_date
        processed_windows += 1
        _emit_progress(current_test_date)
        test_date += timedelta(days=1)

    if total_windows == 0:
        if progress_callback is not None:
            progress_callback(1.0)
    else:
        _emit_progress(test_date)
    
    while test_date <= last_test_date:
        if cancel_event is not None and cancel_event.is_set():
            raise RunAborted("Run aborted by user")

        train_end = test_date - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)

        # Slice data
        train_data = df_work.loc[train_start:train_end - timedelta(minutes=15)].copy()
        test_data = df_work.loc[test_date:test_date + timedelta(days=1) - timedelta(minutes=15)].copy()

        test_data_raw = test_data.copy()

        if train_data.empty or test_data.empty:
            _advance_window(test_date)
            continue

        # Optional preprocessing
        if run_prefiltering:
            for col in forecast_cols:
                B = test_data[col].copy().reset_index(drop=True)
                B_hist = train_data[col].copy().reset_index(drop=True)
                B_proc, B_hist_proc = preprocess_forecast_column(B, B_hist, criteria_dict)
                train_data[col] = B_hist_proc.values
                test_data[col] = B_proc.values

        # Build target series (single or hybrid benchmark).
        if use_secondary_benchmark:
            max_target_date = get_latest_available_target_date_df(test_date, cutoff_df)
            if max_target_date is None:
                _advance_window(test_date)
                continue

            hybrid_target = train_data[primary_target_col].copy()
            hybrid_target.loc[train_data.index > max_target_date] = train_data.loc[
                train_data.index > max_target_date,
                secondary_target_col,
            ]
            X_train_current = train_data[forecast_cols]
            y_train_current = hybrid_target
        else:
            X_train_current = train_data[forecast_cols]
            y_train_current = train_data[primary_target_col]

        X_test = test_data[forecast_cols]
        y_test = test_data[primary_target_col]

        # Skip window if there is no trainable data.
        if X_train_current.empty or y_train_current.empty:
            _advance_window(test_date)
            continue

        invalid_training_data = (
            contains_invalid_values(X_train_current)
            or contains_invalid_values(y_train_current)
        )
        invalid_test_features = contains_invalid_values(X_test)

        # Fit ElasticNet model if enabled and data is valid.
        delnet_predictions = None
        delnet_coeffs = None
        delnet_model = None
        if run_delnet:
            if invalid_training_data or invalid_test_features:
                delnet_predictions = np.full(len(X_test), np.nan)
                delnet_coeffs = np.full(len(forecast_cols), np.nan)
            else:
                delnet_model = ElasticNetCV(
                    l1_ratio=delnet_l1_ratio,
                    eps=delnet_eps,
                    n_alphas=delnet_n_alphas,
                    fit_intercept=delnet_fit_intercept,
                    precompute=delnet_precompute,
                    max_iter=delnet_max_iter,
                    tol=delnet_tol,
                    cv=delnet_cv,
                    copy_X=delnet_copy_X,
                    verbose=delnet_verbose,
                    n_jobs=delnet_n_jobs,
                    positive=delnet_positive,
                    random_state=delnet_random_state,
                    selection=delnet_selection,
                ).fit(X_train_current, y_train_current)
                delnet_predictions = delnet_model.predict(X_test)
                delnet_coeffs = delnet_model.coef_

        # Simple average baseline
        X_test_raw = test_data_raw[forecast_cols]
        y_pred_mean = X_test_raw.mean(axis=1)

        # Individual forecast RMSE
        individual_forecast_rmse = {
            col: safe_rmse(y_test, test_data_raw[col])
            for col in forecast_cols
            if col in test_data_raw.columns
        }

        # Fit PSO ensemble if enabled.
        pso_predictions = None
        pso_weights = None
        if run_pso:
            variant_name, objective_fn = pso_variant
            spread_for_pso = None
            if objective_fn == "balancing_cost_error":
                if not price_spread_data_available:
                    raise ValueError(
                        "Price spread data is required for PSO balancing_cost_error objective function, "
                        "but day_ahead_df and intraday_df are not both available."
                    )

                if price_spread_df is None or "Price_Spread" not in price_spread_df.columns:
                    raise ValueError(
                        "Price spread column not found in merged price data but required for "
                        "PSO balancing_cost_error objective function."
                    )
                spread_for_pso = price_spread_df.loc[
                    X_train_current.index, "Price_Spread"
                ].fillna(0.0).values

            if invalid_training_data or invalid_test_features:
                pso_predictions = np.full(len(X_test), np.nan)
                pso_weights = np.full(len(forecast_cols), np.nan)
            else:
                pso_predictions, pso_weights = _run_pso(
                    X_train_current,
                    y_train_current,
                    X_test,
                    n_features=len(forecast_cols),
                    objective_function=objective_fn,
                    price_spread=spread_for_pso,
                    pso_params=pso_params_cfg,
                )

        # Calculate RMSE for benchmarks
        rmse_mean = safe_rmse(y_test, y_pred_mean)
        # Append results
        df_result = test_data.copy()
        if run_delnet:
            df_result["Pred_DelNet"] = delnet_predictions
            for coeff_idx, coeff_val in enumerate(delnet_coeffs, start=1):
                df_result[f"DelNet_Coeff_F{coeff_idx}"] = coeff_val
        
        df_result["Pred_SA"] = y_pred_mean
        
        if run_pso:
            variant_name, _ = pso_variant
            df_result[f"Pred_{variant_name}"] = pso_predictions
            for weight_idx, weight_val in enumerate(pso_weights, start=1):
                df_result[f"{variant_name}_W{weight_idx}"] = weight_val

        # Add capacity-normalized copies of the target/prediction columns.
        if capacity_series is not None:
            window_capacity = capacity_series.reindex(df_result.index).replace(0, np.nan)
            normalizable_cols = [primary_target_col, "Pred_SA"] + list(forecast_cols)
            if run_delnet:
                normalizable_cols.append("Pred_DelNet")
            if run_pso:
                normalizable_cols.append(f"Pred_{pso_variant[0]}")
            for col in normalizable_cols:
                if col in df_result.columns:
                    df_result[f"{col}_normalized"] = df_result[col] / window_capacity

        results.append(df_result)

        # Compute requested SHAP values for this window.
        if run_delnet and shap_enabled and delnet_model is not None:
            window_shap_df = compute_window_shap_results(
                model=delnet_model,
                X_train=X_train_current,
                y_train=y_train_current,
                X_test=X_test,
                y_test=y_test,
                y_pred=delnet_predictions,
                window_index=i,
                test_date=test_date,
                target_name="DelNet",
                shap_cfg=shap_cfg,
            )
            if not window_shap_df.empty:
                shap_results.append(window_shap_df)

        # Record RMSE metrics
        row = {
            "i": i,
            "window_start": train_start.date(),
            "window_end": train_end.date(),
            "test_date": test_date.date(),
            "RMSE_SA": rmse_mean,
        }
        for forecast_name, forecast_rmse in individual_forecast_rmse.items():
            row[f"RMSE_Forecast_{forecast_name}"] = forecast_rmse

        progress_parts = [f"✓ {i:03d}", f"Test: {test_date.date()}"]

        if run_delnet:
            rmse_delnet = safe_rmse(y_test, delnet_predictions)
            row["RMSE_DelNet"] = rmse_delnet
            progress_parts.append(f"RMSE DelNet: {rmse_delnet:.4f}" if not np.isnan(rmse_delnet) else "RMSE DelNet: NaN")

        progress_parts.append(f"RMSE SA: {rmse_mean:.4f}")

        if run_pso:
            variant_name, _ = pso_variant
            rmse_pso = safe_rmse(y_test, pso_predictions)
            row[f"RMSE_{variant_name}"] = rmse_pso
            progress_parts.append(f"RMSE {variant_name}: {rmse_pso:.4f}" if not np.isnan(rmse_pso) else f"RMSE {variant_name}: NaN")

        # Add capacity-normalized copies of every RMSE column for this window.
        if capacity_series is not None:
            window_capacity_value = capacity_series.reindex(test_data.index).mean()
            if pd.notna(window_capacity_value) and window_capacity_value != 0:
                for rmse_key in [key for key in row if key.startswith("RMSE_")]:
                    row[f"{rmse_key}_normalized"] = row[rmse_key] / window_capacity_value

        rmse_records.append(row)
        print(" | ".join(progress_parts))

        i += 1
        _advance_window(test_date)

    end = time.time()
    runtime = end - start
    print(f"Runtime: {runtime:.4f} seconds")

    # Final summary
    rmse_df = pd.DataFrame(rmse_records)
    df_all_results = pd.concat(results) if results else pd.DataFrame()
    shap_df = pd.concat(shap_results, ignore_index=True) if shap_results else pd.DataFrame()

    print("\n=== Final RMSE Averages ===")
    if not rmse_df.empty:
        rmse_cols = [col for col in rmse_df.columns if col.startswith("RMSE_")]
        for rmse_col in rmse_cols:
            print(f"{rmse_col}: {rmse_df[rmse_col].mean():.4f}")

    return df_all_results, rmse_df, shap_df
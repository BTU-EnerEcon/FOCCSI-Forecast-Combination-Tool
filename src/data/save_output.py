"""Persist pipeline outputs to Excel/config."""

import json
import os

import pandas as pd


def _is_testing_enabled(config):
    """Support both boolean and dict-style testing settings."""
    testing_cfg = config.get("testing", False)
    if isinstance(testing_cfg, dict):
        return bool(testing_cfg.get("enabled", False))
    return bool(testing_cfg)


def _to_json_safe(value):
    """Convert config values (including pandas objects) to JSON-serializable forms."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(col) for col in value.columns],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "shape": [int(value.shape[0])],
            "name": str(value.name),
        }
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_output(    output_data_path,    data_processed,    data_rmse,
    config,
    training_days,
    rmse_outputs=None,
    economic_outputs=None,
    shap_outputs=None,
):
    """Persist pipeline outputs to Excel and save the configuration snapshot."""
    output_path = os.path.join(output_data_path, f"{training_days}_days_training")
    os.makedirs(output_path, exist_ok=True)

    existing_dirs = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]
    test_dirs = [name for name in existing_dirs if name.startswith("test_")]
    result_dirs = [name for name in existing_dirs if name.startswith("results_")]

    testing_enabled = _is_testing_enabled(config)
    if testing_enabled:
        dict_name = f"test_{len(test_dirs) + 1}"
        results_name = f"Test_Model_Results_{training_days}td.xlsx"
    else:
        dict_name = f"results_{len(result_dirs) + 1}"
        results_name = f"Model_Results_{training_days}td.xlsx"

    output_path_this_run = os.path.join(output_path, dict_name)
    os.makedirs(output_path_this_run, exist_ok=False)

    coeff_cols = [
        col_name
        for col_name in data_rmse.columns
        if ("_Coeff_" in col_name) or ("_W" in col_name)
    ]
    model_coeffs_df = (
        data_rmse[["i", "window_start", "window_end", "test_date", *coeff_cols]].copy()
        if coeff_cols
        else pd.DataFrame()
    )

    # Main results workbook.
    with pd.ExcelWriter(os.path.join(output_path_this_run, results_name)) as writer:
        data_rmse.to_excel(writer, sheet_name="RMSE_Summary", index=False)
        data_processed.to_excel(writer, sheet_name="All_Forecasts", index=True)
        if not model_coeffs_df.empty:
            model_coeffs_df.to_excel(writer, sheet_name="Model_Coefficients", index=False)
        if rmse_outputs:
            for sheet_name, df in rmse_outputs.items():
                if df is None or df.empty:
                    continue
                safe_sheet_name = str(sheet_name)[:31]
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        if economic_outputs:
            for sheet_name, df in economic_outputs.items():
                if df is None or df.empty:
                    continue
                safe_sheet_name = str(sheet_name)[:31]
                df.to_excel(writer, sheet_name=safe_sheet_name, index=True)
        if shap_outputs is not None and not shap_outputs.empty:
            shap_outputs.to_excel(writer, sheet_name="SHAP_Values", index=False)

    # Store effective run configuration alongside outputs.
    config_copy = _to_json_safe(dict(config))
    config_copy["training_days"] = int(training_days)
    with open(os.path.join(output_path_this_run, "result_config.txt"), "w", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(config_copy, indent=2))

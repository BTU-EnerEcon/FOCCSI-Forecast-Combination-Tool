"""Pipeline orchestrator: merge inputs, run rolling models, and save outputs."""

import os
import sys
import time
import pandas as pd
# Make src importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.rolling_window_calc import rolling_window_of_timeseries, RunAborted
from src.postprocessing.economic_analysis import compute_economic_analysis
from src.postprocessing.rmse_aggregation import compute_rmse_aggregations
from src.utils.data_processing_utils import merge_input_dataframes
from src.data.save_output import save_output


def run_foccsi(config, progress_callback=None, status_callback=None, cancel_event=None):
    """Run rolling-window training and persist outputs.

    Args:
        config (dict): Configuration dictionary.
        progress_callback (callable, optional): Function to report progress.
        status_callback (callable, optional): Function to report status messages.
        cancel_event (threading.Event, optional): Event to signal run cancellation.

    Raises:
        RunAborted: If the run is aborted by the user.
    """
    start = time.time()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Resolve output folder.
    output_cfg = config.get("output", {})
    output_path_cfg = output_cfg.get("path", "data/output")
    output_data_path = output_path_cfg if os.path.isabs(output_path_cfg) else os.path.join(project_root, output_path_cfg)

    # Precompute controls that do not change per training window.
    rmse_aggregation_enabled = bool(config.get("postprocessing", {}).get("compute_rmse", False))
    economic_analysis_enabled = bool(config.get("economic_analysis", False))
    input_data_cfg = config.get("input_data", {})
    dataset_df, source_column_names, time_col = merge_input_dataframes(input_data_cfg)

    # set the testing configuration for testing mode
    testing_cfg = config.get("testing", False)
    if isinstance(testing_cfg, dict):
        testing_enabled = bool(testing_cfg.get("enabled", False))
        first_test_date_cfg = testing_cfg.get("first_test_date")
        last_test_date_cfg = testing_cfg.get("last_test_date")
    else:
        testing_enabled = bool(testing_cfg)
        first_test_date_cfg = None
        last_test_date_cfg = None

    if testing_enabled and first_test_date_cfg and last_test_date_cfg:
        first_test_date = pd.to_datetime(first_test_date_cfg)
        last_test_date = pd.to_datetime(last_test_date_cfg)
    else:
        first_test_date = None
        last_test_date = None

    # get columns names of input data
    forecast_cols = source_column_names.get("forecasts_df", [])
    benchmark_cols = source_column_names.get("benchmark_df", [])
    secondary_benchmark_cols = source_column_names.get("secondary_benchmark_df", [])

    if not forecast_cols:
        raise ValueError("No forecast columns found for dataset.")

    training_days_list = config.get("rolling_window", {}).get("training_days", [])
    if not training_days_list:
        raise ValueError("No training_days configured under rolling_window.")

    # Iterate over requested train-window lengths.
    steps = len(training_days_list)
    for i, training_days in enumerate(training_days_list):
        if cancel_event is not None and cancel_event.is_set():
            raise RunAborted("Run aborted by user")
        # update status:
        progress_base = i / steps
        progress_span = 1 / steps
        if progress_callback is not None:
            progress_callback(progress_base)

        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (steps - i - 1)

        if status_callback is not None:
            status_callback(f"Step {i+1}/{steps} | ETA: {eta:.1f} sec")

        def _window_progress(local_progress, progress_base=progress_base, progress_span=progress_span):
            # Map inner rolling progress (0..1) to global run progress.
            mapped_progress = progress_base + (progress_span * float(local_progress))
            if progress_callback is not None:
                progress_callback(min(1.0, max(0.0, mapped_progress)))

        def _window_status(message, i=i, steps=steps):
            if status_callback is not None:
                status_callback(f"Step {i+1}/{steps} | {message}")

        # start the rolling window calculation
        data_processed, data_rmse, data_shap = rolling_window_of_timeseries(
            df=dataset_df,
            time_col=time_col,
            forecast_cols=forecast_cols,
            benchmark_cols=benchmark_cols,
            secondary_benchmark_cols=secondary_benchmark_cols,
            train_days=training_days,
            test_days=config['rolling_window'].get('horizon', 1),
            model_config=config,
            progress_callback=_window_progress,
            status_callback=_window_status,
            cancel_event=cancel_event,
            first_test_date=first_test_date,
            last_test_date=last_test_date
        )

        rmse_outputs = None
        if rmse_aggregation_enabled:
            # Optional calendar aggregation of RMSE metrics.
            rmse_outputs = compute_rmse_aggregations(data_rmse, config)

        economic_outputs = None
        if economic_analysis_enabled:
            economic_outputs = compute_economic_analysis(
                data_processed,
                forecast_cols,
                benchmark_cols,
                input_data_cfg,
                config,
            )
        # save output 
        save_output(
            output_data_path=output_data_path,
            data_processed=data_processed,
            data_rmse=data_rmse,
            config=config,
            training_days=training_days,
            rmse_outputs=rmse_outputs,
            economic_outputs=economic_outputs,
            shap_outputs=data_shap,
        )

    if progress_callback is not None:
        progress_callback(1.0)
    return {"status": "completed", "message": "FOCCSI run finished"}


if __name__ == "__main__":
    raise SystemExit("Use scripts/RUN_app.py or scripts/run_app_debug_temp.py to run the pipeline.")
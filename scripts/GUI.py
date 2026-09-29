"""Streamlit UI for configuring and running the FOCCSI pipeline."""

import streamlit as st
import os
import sys
import threading
import time
from datetime import date, timedelta
# Make src importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.loader import load_csv
from src.data.analyze_input import analyze_uploaded_data
from src.data.analyze_input import analyze_selected_config
from run_main import RunAborted, run_foccsi


def _init_session_state():
    """Initialize all upload placeholders used across the app tabs."""
    # Central list of optional/required uploads used across all tabs.
    upload_keys = [
        "forecasts_df",
        "benchmark_df",
        "secondary_benchmark_df",
        "secondary_benchmark_cutoff_df",
        "day_ahead_df",
        "intraday_df",
        "installed_capacity_df",
    ]
    for key in upload_keys:
        if key not in st.session_state:
            st.session_state[key] = None


st.set_page_config(page_title="FOCCSI Forecast Combination", layout="wide")
_init_session_state()

st.title("FOCCSI – Forecast Combination Tool")

tab1, tab2, tab3, tab4 = st.tabs(["Task Definition", "Input Data Analysis", "Model Parameter Configuration", "Run"])

# Tab 1: upload raw input datasets into session state.
with tab1:
    st.header("Task Definition")
    st.subheader("Choose Combination Model")
    st.selectbox(
        "Elastic Net (DELNET)",
        ["Yes", "No"],
        index=0,
        key="run_delnet_select",
    )
    run_pso_selection = st.selectbox(
        "Particle Swarm Optimization (PSO)",
        ["No", "Yes"],
        index=0,
        key="run_pso_select",
    )
    if run_pso_selection == "Yes":
        pso_objective_label = st.selectbox(
            "Select PSO objective function",
            [
                "Root Mean Squared Error (RMSE)",
                "Balancing Price Error Optimized",
            ],
            index=0,
            key="pso_objective_select",
        )

    st.subheader("Forecast Energy Time Series")
    forecast_file = st.file_uploader("Forecast Energy Data", type=["csv"], key="forecast_file")
    if st.button("Upload Forecast Data"):
        if forecast_file is not None:
            st.session_state["forecasts_df"] = load_csv(forecast_file)
            st.success("Forecast file uploaded!")
        else:
            st.info("No forecast file provided.")

    st.subheader("Benchmark Energy Timeseries Data")
    benchmark_file = st.file_uploader("Benchmark Energy Data", type=["csv"], key="benchmark_file")
    if st.button("Upload Benchmark Data"):
        if benchmark_file is not None:
            st.session_state["benchmark_df"] = load_csv(benchmark_file)
            st.success("Benchmark file uploaded!")
        else:
            st.info("No benchmark file provided.")

    use_secondary_benchmark = st.selectbox(
        "Use secondary benchmark?",
        ["No", "Yes"],
        index=0,
        key="use_secondary_benchmark",
    )
    if use_secondary_benchmark == "Yes":
        secondary_benchmark_file = st.file_uploader(
            "Secondary Benchmark Energy Data",
            type=["csv"],
            key="secondary_benchmark_file",
        )
        secondary_benchmark_cutoff_file = st.file_uploader(
            "Last available dates for benchmark energy data",
            type=["csv"],
            key="secondary_benchmark_cutoff_file",
        )
        if st.button("Upload Secondary Benchmark Data"):
            if secondary_benchmark_file is not None:
                st.session_state["secondary_benchmark_df"] = load_csv(secondary_benchmark_file)
                st.success("Secondary benchmark file uploaded!")
            else:
                st.info("No secondary benchmark file provided.")

            if secondary_benchmark_cutoff_file is not None:
                st.session_state["secondary_benchmark_cutoff_df"] = load_csv(secondary_benchmark_cutoff_file)
                st.success("Last available dates file uploaded!")
            else:
                st.info("No last available dates file provided.")
    else:
        st.session_state["secondary_benchmark_df"] = None
        st.session_state["secondary_benchmark_cutoff_df"] = None

    normalize_output_data = st.selectbox(
        "Normalize output data?",
        ["No", "Yes"],
        index=0,
        key="normalize_output_data",
    )
    if normalize_output_data == "Yes":
        installed_capacity_file = st.file_uploader(
            "Installed Capacity Data",
            type=["csv"],
            key="installed_capacity_file",
        )
        if st.button("Upload Installed Capacity Data"):
            if installed_capacity_file is not None:
                st.session_state["installed_capacity_df"] = load_csv(installed_capacity_file)
                st.success("Installed capacity file uploaded!")
            else:
                st.info("No installed capacity file provided.")
    else:
        st.session_state["installed_capacity_df"] = None

    st.subheader("Post Processing")
    calculate_rmse_aggregation = st.selectbox(
        "Calculate Root Mean Square Error (RMSE) Aggregation",
        ["No", "Yes"],
        index=0,
        key="calculate_rmse_aggregation",
    )
    if calculate_rmse_aggregation == "Yes":
        st.multiselect(
            "RMSE aggregation levels",
            ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
            default=["hourly", "daily"],
            help="Select one or more levels for RMSE aggregation.",
            key="rmse_aggregation_levels",
        )
    else:
        st.session_state["rmse_aggregation_levels"] = []

    st.selectbox(
        "Calculate Shapley Values (perfomance based, reduced game, zero coalition ignored)",
        ["Yes", "No"],
        index=0,
        key="shap_calc_select",
    )

    run_economic_analysis = st.selectbox(
        "Run economic error analysis",
        ["No", "Yes"],
        index=0,
        key="run_economic_analysis",
    )

    price_data_needed = run_economic_analysis == "Yes" or (
        run_pso_selection == "Yes"
        and pso_objective_label == "Balancing Price Error Optimized"
    )
    if price_data_needed:
        st.caption("Upload Energy Prices Data")
        day_ahead_file = st.file_uploader("Day-Ahead Prices", type=["csv"], key="day_ahead_file")
        intraday_file = st.file_uploader("Intraday Prices", type=["csv"], key="intraday_file")
        if st.button("Upload Energy Prices Data"):
            if day_ahead_file is not None:
                st.session_state["day_ahead_df"] = load_csv(day_ahead_file)
                st.success("Day-ahead prices file uploaded!")
            else:
                st.info("No day-ahead prices file provided.")

            if intraday_file is not None:
                st.session_state["intraday_df"] = load_csv(intraday_file)
                st.success("Intraday prices file uploaded!")
            else:
                st.info("No intraday prices file provided.")

    if run_economic_analysis == "Yes":
        st.multiselect(
            "Economic cost aggregation levels",
            ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
            default=["monthly", "yearly"],
            help="Balancing costs are summed within each selected period.",
            key="economic_aggregation_levels",
        )
    else:
        st.session_state["economic_aggregation_levels"] = []

# Tab 2: run exploratory quality checks for uploaded datasets.
with tab2:
    st.header("Input Data Analysis")

    uploaded_datasets = {
        "Forecast Data": st.session_state.forecasts_df,
        "Benchmark Data": st.session_state.benchmark_df,
        "Secondary Benchmark Data": st.session_state.secondary_benchmark_df,
        "Secondary Benchmark Cutoff Data": st.session_state.secondary_benchmark_cutoff_df,
        "Day-Ahead Prices": st.session_state.day_ahead_df,
        "Intraday Prices": st.session_state.intraday_df,
        "Installed Capacity": st.session_state.installed_capacity_df,
    }

    available_datasets = {
        name: df for name, df in uploaded_datasets.items() if df is not None
    }

    if available_datasets:
        if st.button("Analyze Input Data"):
            for dataset_name, dataset_df in available_datasets.items():
                analysis = analyze_uploaded_data(dataset_df)

                st.subheader(dataset_name)

                if "error" in analysis:
                    st.warning(analysis["error"])
                    continue

                st.write("**Detected Columns**")
                st.write(analysis["columns"])

                st.write("**Column Summary (0-values, mean, max, min)**")
                st.dataframe(analysis["column_summary"], use_container_width=True)

                st.write("**Time Range**")
                st.write(f"First timestamp: {analysis['start']}")
                st.write(f"Last timestamp: {analysis['end']}")

                st.write("**Time Frequency**")
                st.write(analysis["frequency"])

                st.write("**Missing Values per Column**")
                st.write(analysis["missing_values"])

                st.write("**Missing Timestamp Periods (dataset-level)**")
                missing_timestamp_periods = analysis["missing_periods"]
                if missing_timestamp_periods:
                    for period in missing_timestamp_periods:
                        st.write(
                            f"Between {period['start']} and {period['end']}: "
                            f"{period['missing_steps']} missing timestamps"
                        )
                else:
                    st.write("No missing time periods detected.")

                st.write("**Missing Data Periods per Non-Time Column**")
                missing_by_column = analysis["missing_periods_by_column"]
                value_columns = analysis["value_columns"]

                if not value_columns:
                    st.write("No non-time columns available.")
                else:
                    for col in value_columns:
                        st.write(f"Column: {col}")
                        col_periods = missing_by_column.get(col, [])
                        if col_periods:
                            for period in col_periods:
                                st.write(
                                    f"Between {period['start']} and {period['end']}: "
                                    f"{period['missing_steps']} missing values"
                                )
                        else:
                            st.write("No missing periods for this column.")

                st.divider()
    else:
        st.info("Upload at least one file in Tab 1 to enable analysis.")

# Tab 3: configure model and preprocessing parameters.
with tab3:
    st.header("Model Parameter Configuration")

    run_delnet = st.session_state.get("run_delnet_select", "Yes") == "Yes"
    run_pso = st.session_state.get("run_pso_select", "No") == "Yes"

    pso_params = {
        "swarm_size": 30,
        "iterations": 50,
        "decay": 0,
        "w": 0.9,
        "c1": 2,
        "c2": 2,
    }
    if run_pso:
        st.subheader("PSO Settings")
        pso_param_mode = st.radio(
            "Parameter mode",
            ["Standard parameters", "Manual parameters"],
            index=0,
            key="pso_param_mode",
        )
        if pso_param_mode == "Manual parameters":
            col1, col2, col3 = st.columns(3)
            with col1:
                pso_params["swarm_size"] = st.number_input("Swarm size", min_value=1, value=30, step=1)
                pso_params["iterations"] = st.number_input("Iterations", min_value=1, value=50, step=1)
            with col2:
                pso_params["decay"] = st.number_input("Decay", min_value=0, value=0, step=1)
                pso_params["w"] = st.number_input("Inertia weight (w)", min_value=0.0, value=0.9, step=0.05)
            with col3:
                pso_params["c1"] = st.number_input("Cognitive coefficient (c1)", min_value=0.0, value=2.0, step=0.1)
                pso_params["c2"] = st.number_input("Social coefficient (c2)", min_value=0.0, value=2.0, step=0.1)

    if run_delnet:
        st.subheader("ElasticNet Settings")
        elasticnet_param_mode = st.radio(
            "Parameter mode",
            ["Standard parameters", "Manual parameters"],
            index=0,
            key="elasticnet_param_mode",
        )

        if elasticnet_param_mode == "Standard parameters":
            l1_ratio = 0.5
            cv = 10
            elasticnet_eps = 0.001
            elasticnet_n_alphas = 100
            elasticnet_fit_intercept = True
            elasticnet_precompute = "auto"
            elasticnet_max_iter = 1000
            elasticnet_tol = 0.0001
            elasticnet_copy_x = True
            elasticnet_verbose = 0
            elasticnet_n_jobs = None
            elasticnet_positive = False
            elasticnet_random_state = None
            elasticnet_selection = "cyclic"
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                l1_ratio = st.number_input("L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                elasticnet_eps = st.number_input(
                    "eps (alpha_min / alpha_max)", min_value=0.0, value=0.001, step=0.0001, format="%.6f"
                )
                elasticnet_max_iter = st.number_input("Max iterations", min_value=1, value=1000, step=100)
                elasticnet_verbose = st.number_input("Verbosity", min_value=0, value=0, step=1)
            with col2:
                cv = st.number_input("Cross Validation Folds", min_value=2, value=10, step=1)
                elasticnet_n_alphas = st.number_input(
                    "Number of alphas (n_alphas)", min_value=1, value=100, step=1
                )
                elasticnet_tol = st.number_input("Tolerance", min_value=0.0, value=0.0001, step=0.0001, format="%.6f")
                set_n_jobs = st.checkbox("Set n_jobs", value=False)
                elasticnet_n_jobs = st.number_input("n_jobs", value=-1, step=1) if set_n_jobs else None
            with col3:
                precompute_label = st.selectbox("Precompute", ["auto", "True", "False"], index=0)
                elasticnet_precompute = {"auto": "auto", "True": True, "False": False}[precompute_label]
                elasticnet_selection = st.selectbox("Selection", ["cyclic", "random"], index=0)
                elasticnet_fit_intercept = st.checkbox("Fit intercept", value=True)
                elasticnet_copy_x = st.checkbox("Copy X", value=True)
                elasticnet_positive = st.checkbox("Force positive coefficients", value=False)
                set_random_state = st.checkbox("Set random_state", value=False)
                elasticnet_random_state = st.number_input("random_state", min_value=0, value=0, step=1) if set_random_state else None
    else:
        l1_ratio = 0.5
        cv = 10
        elasticnet_eps = 0.001
        elasticnet_n_alphas = 100
        elasticnet_fit_intercept = True
        elasticnet_precompute = "auto"
        elasticnet_max_iter = 1000
        elasticnet_tol = 0.0001
        elasticnet_copy_x = True
        elasticnet_verbose = 0
        elasticnet_n_jobs = None
        elasticnet_positive = False
        elasticnet_random_state = None
        elasticnet_selection = "cyclic"

    st.subheader("Rolling Window")
    training_days = st.multiselect(
        "Training Days",
        [30, 60, 90, 120, 180],
        default=[90]
    )

    forecast_horizon = st.number_input("Training Frequency (days)", value=1)

    st.subheader("Data Preprocessing")

    filter_benchmark_data = st.checkbox("Filter benchmark data", value=False)
    benchmark_threshold = None
    if filter_benchmark_data:
        benchmark_threshold = st.number_input(
            "Benchmark filter threshold",
            value=0.0,
            step=0.1,
        )

    filter_forecast_data = st.checkbox("Filter forecast data", value=False)
    forecast_thresholds = {}
    if filter_forecast_data:
        st.caption("Training window thresholds")
        forecast_thresholds["training_window"] = {
            "HIV": st.number_input("HIV - max negative values in B_hist", value=100, step=1),
            "HNN": st.number_input("HNN - max consecutive NaNs in B_hist", value=96, step=1),
            "HRN": st.number_input("HRN - max total NaNs in B_hist", value=960, step=1),
            "HRZ": st.number_input("HRZ - max consecutive repeated zeros in B_hist", value=960, step=1),
            "HRV": st.number_input("HRV - max consecutive repeated non-zero values in B_hist", value=960, step=1),
        }

        st.caption("Horizon window thresholds")
        forecast_thresholds["horizon_window"] = {
            "IV": st.number_input("IV - max negative values in B", value=5, step=1),
            "NN": st.number_input("NN - max consecutive NaNs in B", value=5, step=1),
            "RN": st.number_input("RN - max total NaNs in B", value=10, step=1),
            "RZ": st.number_input("RZ - max consecutive repeated zeros in B", value=68, step=1),
            "RV": st.number_input("RV - max consecutive repeated non-zero values in B", value=10, step=1),
        }

# Tab 4: build run config, validate it, and execute the pipeline.
with tab4:
    st.header("Run FOCCSI Pipeline")

    forecasts_df = st.session_state.forecasts_df
    benchmark_df = st.session_state.benchmark_df
    secondary_benchmark_df = st.session_state.secondary_benchmark_df
    secondary_benchmark_cutoff_df = st.session_state.secondary_benchmark_cutoff_df
    day_ahead_df = st.session_state.day_ahead_df
    intraday_df = st.session_state.intraday_df
    installed_capacity_df = st.session_state.installed_capacity_df

    shap_calc = st.session_state.get("shap_calc_select", "Yes") == "Yes"

    st.subheader("Output Settings")
    output_path = st.text_input(
        "Output path",
        value="data/output",
        help="Relative paths are resolved from the project root.",
    )

    st.subheader("Testing")
    test_data_mode = st.selectbox(
        "Test data selection",
        ["Use all provided data", "Select test dates in range"],
        index=0,
        key="test_data_mode",
    )
    testing_enabled = test_data_mode == "Select test dates in range"

    first_test_date = None
    last_test_date = None
    if testing_enabled:
        default_test_date_range = (date.today() - timedelta(days=30), date.today())
        if forecasts_df is not None and "time" in forecasts_df.columns:
            forecast_timestamps = forecasts_df["time"].dropna()
            if not forecast_timestamps.empty:
                default_test_date_range = (
                    forecast_timestamps.min().date(),
                    forecast_timestamps.max().date(),
                )

        with st.popover("Select test date range"):
            selected_range = st.date_input(
                "Test date range",
                value=default_test_date_range,
                key="test_date_range",
            )
        if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
            first_test_date, last_test_date = selected_range
        elif isinstance(selected_range, date):
            first_test_date = last_test_date = selected_range

    compute_rmse = st.session_state.get("calculate_rmse_aggregation", "No") == "Yes"
    rmse_aggregation_levels = st.session_state.get("rmse_aggregation_levels", [])
    economic_analysis = st.session_state.get("run_economic_analysis", "No") == "Yes"
    normalize_output_enabled = st.session_state.get("normalize_output_data", "No") == "Yes"
    pso_objective_label = st.session_state.get(
        "pso_objective_select", "Root Mean Squared Error (RMSE)"
    )
    pso_objective = (
        "rmse"
        if pso_objective_label.startswith("Root Mean")
        else "balancing_price_error"
    )
    pso_requires_prices = run_pso and pso_objective == "balancing_price_error"


    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False

    def _start_pipeline_thread(run_config):
        """Launch run_foccsi on a background thread so the UI stays responsive."""
        cancel_event = threading.Event()
        state = {"progress": 0.0, "status": "Starting...", "done": False, "error": None, "result": None}

        def _worker():
            try:
                run_result = run_foccsi(
                    run_config,
                    progress_callback=lambda p: state.update(progress=p),
                    status_callback=lambda msg: state.update(status=msg),
                    cancel_event=cancel_event,
                )
                state["result"] = run_result
            except RunAborted:
                state["error"] = "Run aborted by user."
            except Exception as exc:
                state["error"] = f"Run failed: {exc}"
            finally:
                state["done"] = True

        st.session_state["pipeline_cancel_event"] = cancel_event
        st.session_state["pipeline_state"] = state
        st.session_state["pipeline_running"] = True
        threading.Thread(target=_worker, daemon=True).start()

    if not st.session_state["pipeline_running"]:
        if st.button("START RUN"):
            # Collect all uploaded datasets into the runtime config.
            input_data = {
                "forecasts_df": forecasts_df,
                "benchmark_df": benchmark_df,
                "secondary_benchmark_df": secondary_benchmark_df,
                "secondary_benchmark_cutoff_df": secondary_benchmark_cutoff_df,
                "day_ahead_df": day_ahead_df,
                "intraday_df": intraday_df,
                "installed_capacity_df": installed_capacity_df,
            }

            config = {
                "models": {
                    "delnet": run_delnet,
                    "pso": run_pso,
                    "pso_objective": pso_objective,
                    "pso_params": {
                        "swarm_size": int(pso_params["swarm_size"]),
                        "iterations": int(pso_params["iterations"]),
                        "decay": pso_params["decay"],
                        "w": pso_params["w"],
                        "c1": pso_params["c1"],
                        "c2": pso_params["c2"],
                    },
                    "elasticnet_params": {
                        "l1_ratio": l1_ratio,
                        "cv": int(cv),
                        "eps": elasticnet_eps,
                        "n_alphas": int(elasticnet_n_alphas),
                        "fit_intercept": elasticnet_fit_intercept,
                        "precompute": elasticnet_precompute,
                        "max_iter": int(elasticnet_max_iter),
                        "tol": elasticnet_tol,
                        "copy_X": elasticnet_copy_x,
                        "verbose": int(elasticnet_verbose),
                        "n_jobs": int(elasticnet_n_jobs) if elasticnet_n_jobs is not None else None,
                        "positive": elasticnet_positive,
                        "random_state": int(elasticnet_random_state) if elasticnet_random_state is not None else None,
                        "selection": elasticnet_selection,
                    },
                    "shap_enabled": shap_calc,
                    "shap_config": {
                        "sv_pb_rg_zci": shap_calc,
                    },
                },
                "rolling_window": {
                    "training_days": training_days,
                    "horizon": forecast_horizon,
                },
                "preprocessing": {
                    "filter_benchmark_data": filter_benchmark_data,
                    "benchmark_threshold": benchmark_threshold,
                    "filter_forecast_data": filter_forecast_data,
                    "forecast_thresholds": forecast_thresholds,
                },
                "output": {
                    "path": output_path,
                },
                "economic_analysis": economic_analysis,
                "postprocessing": {
                    "compute_rmse": compute_rmse,
                    "rmse_aggregation_levels": rmse_aggregation_levels,
                    "economic_aggregation_levels": st.session_state.get("economic_aggregation_levels", []),
                    "normalize_output": normalize_output_enabled,
                },
                "testing": {
                    "enabled": testing_enabled,
                    "first_test_date": first_test_date.isoformat() if first_test_date else None,
                    "last_test_date": last_test_date.isoformat() if last_test_date else None,
                },
                "input_data": input_data,
            }

            if testing_enabled and (first_test_date is None or last_test_date is None):
                st.write("**Selected Configuration Analysis**")
                st.write("Please select a valid test date range.")
            elif normalize_output_enabled and installed_capacity_df is None:
                st.write("**Selected Configuration Analysis**")
                st.write("Normalize output is enabled but no installed capacity data was uploaded (Tab 1).")
            elif (economic_analysis or pso_requires_prices) and (
                day_ahead_df is None or intraday_df is None
            ):
                st.write("**Selected Configuration Analysis**")
                st.write(
                    "Day-ahead and intraday price data must both be uploaded in Tab 1 for "
                    "economic analysis or the balancing-price PSO objective."
                )
            else:
                missing_config, missing_config_message = analyze_selected_config(config)
                if missing_config:
                    st.write("**Selected Configuration Analysis**")
                    st.write(missing_config_message)
                else:
                    st.write("**Selected Configuration Analysis**")
                    st.write("All required configuration parameters are set.")
                    _start_pipeline_thread(config)
                    st.rerun()
    else:
        # A run is in progress: show live progress and an abort control.
        state = st.session_state["pipeline_state"]

        st.progress(min(1.0, max(0.0, state["progress"])))
        st.text(state["status"])

        if st.button("Abort Run"):
            st.session_state["pipeline_cancel_event"].set()
            st.info("Aborting... waiting for the current step to stop.")

        if state["done"]:
            st.session_state["pipeline_running"] = False
            if state["error"]:
                st.error(state["error"])
            else:
                st.success(state["result"]["message"])
        else:
            time.sleep(0.5)
            st.rerun()
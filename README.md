# Forecast Optimisation by Correction and Combination methods for System Integration (FOCCSI)


FOCCSI (Forecast Optimisation by Correction and Combination Methods for System Integration) is a Python-based research framework for the correction, combination, and evaluation of forecasts for highly dynamic time series. The framework was developed in the context of the FOCCSI 2 academic research project, with a particular focus on renewable energy generation and electricity-related forecasting applications.

The main objective of FOCCSI is to investigate whether combining several available forecasts can produce a more accurate and robust meta-forecast than the individual forecasts themselves. The framework supports statistical and machine-learning-based forecast combination methods and evaluates their performance using both forecasting accuracy measures and, where applicable, economic criteria.

The current implementation focuses on two forecast-combination approaches:
-   Dynamic Elastic Net (DELNET): A regularized linear regression approach used to estimate forecast-combination weights. 
- Particle Swarm Optimization (PSO): A swarm-intelligence-based optimization approach that determines forecast-combination weights by minimizing a selected objective function.


The approaches implemented in FOCCSI are motivated by research on renewable-energy feed-in forecasting. In particular, the framework can be applied to time series of solar photovoltaic and wind power forecasts and is designed to support dynamic forecasting environments in which the relationship between individual forecasts and the observed outcome can change over time.

In addition to forecast combination, FOCCSI provides functionality for:
- Forecast correction and preprocessing
- Dynamic forecast combination and ensemble modelling
- Forecast performance evaluation through mean Root Mean Square Deviation (RMSE).
- Result aggregation at different temporal levels
- Performance-based Shapley value analysis
- Optional normalization of forecast output using installed capacity
- Economic forecast-error analysis using electricity prices


The application is operated through a Streamlit graphical user interface (GUI). The GUI guides the user through task selection, data inspection, model configuration, and execution of the FOCCSI pipeline.

## Features
- Forecast correction and preprocessing
- Forecast combination using Dynamic Elastic Net (DELNET)
- Forecast combination using Particle Swarm Optimization (PSO)
- Rolling-window estimation and evaluation
- Configurable model parameters
- Input-data quality analysis
- RMSE calculation and aggregation
- Performance-based Shapley value analysis
- Optional output normalization
- Economic forecast-error analysis
- Configurable test periods
- Graphical user interface based on Streamlit


## Installation

FOCCSI is a Python-based application and requires Python and Git to be installed on the user's system. You can set up the project either with Conda (recommended, uses `requirements.yml`) or with pip and a virtual environment (uses `requirements.txt`).

### Prerequisites

Before installing FOCCSI, make sure that the following software is installed:

- [Git](https://git-scm.com/)
- Either [Anaconda/Miniconda](https://www.anaconda.com/docs/main) or [Python](https://www.python.org/) (recommended version: **Python 3.x**)

You can verify the installations from a terminal:

```bash
git --version
conda --version   # if using Conda
python --version  # if using pip
```



### 1. Clone the repository

Clone the FOCCSI repository to your local machine:

```bash
git clone https://github.com/BTU-EnerEcon/FOCCSI-Forecast-Combination-Tool
```


Then navigate to the project directory:

```bash
cd Code_FOCCSI2
```

### 2. Set up the environment

Choose **one** of the following methods.

#### Option A: Conda (recommended)

Create the environment from the provided `requirements.yml` file, which defines the environment name and all pinned dependencies:

```bash
conda env create -f requirements.yml
conda activate env_foccsi
```

#### Option B: pip + virtual environment

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify the installation

```bash
# Conda
conda list

# pip
pip list
```


## Run the App
After installing/downloading the project, navigate to the `scripts` directory from a Bash terminal:

```bash
cd scripts
```
Start the Streamlit application with:
```bash
streamlit run GUI.py
```

Streamlit will start the FOCCSI web application and normally open it automatically in the user's default web browser.

The application is divided into the tabs:
-  `Task Definition`: select the forecasting tasks and provide the required input data.
- `Input Data Analysis`: inspect the uploaded datasets and identify potential data-quality problems.
-  `Model Parameter Configutation`: configure model parameters, rolling-window settings, and preprocessing options.
- `Run`: validate the selected configuration and execute the FOCCSI pipeline.

The following sections describe how to use each tab.


## Task Definition
The Task Definition tab is the starting point for a FOCCSI analysis. Here, the user specifies which forecast-combination methods and additional analyses should be performed and uploads the corresponding input data.
The application stores the uploaded datasets and selected options and makes them available to the subsequent tabs and the FOCCSI pipeline.

**General Format for required Input data:** \
Input data should be provided as CSV files with the following format:

- File type: `.csv`
- Delimiter: `;` (semicolon)
- Decimal separator: `,`
- Thousands separator: `.`
- Timestamp format: `dd.mm.yyyy HH:MM`

> Important: Make sure that the timestamps of the forecast and benchmark data are aligned

### Choose Combination Model
In the Choose Combination Model section, the user selects which forecast-combination methods should be executed.
FOCCSI currently provides two combination approaches:

**Dynamic Elastic Net (DELNET):**\
DELNET combines the available forecasts by estimating a set of combination coefficients using an Elastic Net regression model. The Elastic Net approach applies both L1 and L2 regularization and uses cross-validation to determine the model parameters.
Within this pipeline, the model is estimated dynamically using a rolling training window. This allows the combination coefficients to adapt over time to changes in the forecasting performance of the individual input forecasts.
DELNET is therefore particularly suited for applications where an interpretable, regression-based forecast combination is desired.

**Particle Swarm Optimization (PSO)**\
PSO combines the available forecasts by directly optimizing their combination weights. The weights are determined through a Particle Swarm Optimization algorithm, which searches for a set of weights that minimizes a predefined objective function. When PSO is selected, the Task Definition tab also displays the objective-function selector:

- `Root Mean Squared Error (RMSE)`: minimizes the RMSE between the combined forecast and the benchmark.
- `Balancing Price Error Optimized`: minimizes the squared forecast error weighted by the price spread between the Intraday and Day-Ahead markets.


The parameters for the selected combination model can be configured in the `Model Parameter Configuration` tab.


### Forecast Energy Time Series

The Forecast Energy Time Series section is used to provide the forecasts that should be combined.

The user uploads a CSV file containing the available forecasts. Each forecast should represent an individual forecasting source/model that will be considered by the combination algorithms.


**Required Forecast Data Structure**
- First row: column headers
- First column: timestamp information (`dd.mm.yyyy MM:HH`)
- Remaining columns: Individual forecast time series 
- Unit:  MW

All forecast series that are intended to be combined should use the same timestamps.
The number of input forecasts can be adapted to the application. However, increasing the number of forecasts can also increase the computational effort required by the combination methods.

The application accepts CSV files through the Forecast Energy Data upload field. After selecting the file, click `Upload Forecast Data`. A successful upload is confirmed by the message:

>Forecast file uploaded!

### Benchmark Energy Timeseries

The Benchmark Energy Timeseries Data section is used to provide the observed/reference energy time series against which the forecasts are evaluated.

The benchmark represents the actual energy generation or feed-in against which the forecast results are compared.


**Required Benchmark Data Structure:**
- First row: column headers
- First column: timestamp information (`dd.mm.yyyy MM:HH`)
- Second column: Benchmark/observed energy value
- Unit: MW

Upload the benchmark CSV file using `Benchmark Energy Data` field and then click Upload `Benchmark Data`. After a successful upload, the application displays:
> Benchmark file uploaded!

`Benchmark Energy Data`:\
This is the main benchmark data


### Secondary Benchmark  (optional)
The toolbox supports a secondary benchmark time series.
This option can be enabled using:\
`Secondary Benchmark Energy Data` → `yes`

When enabled, the user must provide:
- A secondary benchmark energy dataset
- A file containing the last available dates for the benchmark energy data

The secondary benchmark functionality can be used when benchmark information is available from an additional source and the different benchmark periods need to be combined during the analysis. for each training data window the benchmark data will be replaced with the secondary benchmark data for the perion between the test date and the provided last available date
After selecting the files, click `Upload Secondary Benchmark Data`.
 
In energy forecasting often the most accurate realized production data only becomes available once a month. This missing benchmark data can be replaced by an secondary benchmark dataset using this option.

### Normalize Output Data (optional)
The Normalize output data option allows the user to normalize the forecast output using installed-capacity information.

Select:

`Normalize output data` → `Yes`

When normalization is enabled, an Installed Capacity Data file must also be uploaded.

The installed-capacity dataset is used by the pipeline as an additional input for the normalization step.

### Post Processing

The Post Processing section provides additional analysis options after the forecast-combination process.

**RMSE Aggregation**

The user can enable:

`Calculate Root Mean Square Error (RMSE) Aggregation` → `Yes`

When enabled, one or more aggregation levels can be selected (Hourly, Daily, Weekly, Monthly, Quarterly, Yearly)

This allows forecast errors to be evaluated at different temporal aggregation levels.

**Shapley Value Analysis**

The option:

`Calculate Shapley Values`

enables a performance-based Shapley value analysis of the forecast combination.

The current implementation uses a performance-based approach with a reduced game and ignores the zero coalition. The resulting analysis can be used to assess the contribution of individual forecasts to the performance of the combined forecast.

**Economic Error Analysis**

The application can additionally perform an economic analysis of forecast errors.

Enable:

`Run economic error analysis` → `Yes`

The Day-Ahead Prices and Intraday Prices upload fields are shown when economic error analysis is selected, or when PSO uses the `Balancing Price Error Optimized` objective. Uploaded price data remains available for the current app session and can be reused by both features.

Both price datasets are required when economic error analysis is enabled. They are also required when PSO uses the `Balancing Price Error Optimized` objective. Standard PSO with the RMSE objective does not require price data.

Provide:

- Day-ahead electricity prices
- Intraday electricity prices

This analysis extends the evaluation beyond purely physical forecast accuracy and allows forecast errors to be considered from an economic perspective.


## Input Data Analysis
The Input Data Analysis tab provides exploratory data-quality checks for every dataset uploaded in the `Task Definition` tab (forecasts, benchmark, secondary benchmark, secondary benchmark cutoff, day-ahead prices, intraday prices, installed capacity).

Only datasets that have actually been uploaded are analyzed; if nothing has been uploaded yet, the tab shows a hint to upload at least one file in `Task Definition`.

Clicking `Analyze Input Data` generates, per dataset:
- **Detected Columns** – the column names found in the file.
- **Column Summary** – number of zero values, mean, max, and min per column.
- **Time Range** – first and last timestamp in the dataset.
- **Time Frequency** – the detected sampling interval (e.g. 15 minutes).
- **Missing Values per Column** – count of missing (NaN) values per column.
- **Missing Timestamp Periods (dataset-level)** – gaps in the timestamp index, i.e. periods where expected timestamps are absent entirely.
- **Missing Data Periods per Non-Time Column** – gaps in the actual values of each data column (as opposed to gaps in the timestamp index itself).

This tab is intended to be used before configuring or running the pipeline, so that data-quality issues (missing timestamps, missing values, misaligned time ranges) can be identified and corrected early.

## Model Parameter Configuration
The Model Parameter Configuration tab is used to configure the forecast-combination models, the rolling-window settings, and optional data-preprocessing/filtering steps. Only the sections relevant to the models selected in `Task Definition` are shown.

### PSO Settings
If Particle Swarm Optimization (PSO) was selected in `Task Definition`, this section becomes available. 
Further information about the implemented PSO algorithm can be found on the corresponding GitHub page (https://github.com/Valdecy/pyMetaheuristic) of the package used for the implementation.

**PSO Settings**\
The PSO algorithm's own parameters can be left at their standard values or configured manually:

- `Standard parameters` – uses the built-in defaults (swarm size 30, 50 iterations, decay 0, inertia weight `w` 0.9, cognitive/social coefficients `c1`/`c2` = 2).
- `Manual parameters` – exposes swarm size, iterations, decay, inertia weight (`w`), and the cognitive/social coefficients (`c1`, `c2`) for direct editing.

### ElasticNet Settings
If Dynamic Elastic Net (DELNET) is selected in the `Task Definition`, this section becomes available.

The algorithm is implemented using the `scikit-learn` package. Further information about the underlying `ElasticNetCV` implementation and its parameters can be found in the official scikit-learn documentation (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html).

As with PSO, the underlying `ElasticNetCV` parameters can either be left at their standard values or configured manually:

- `Standard parameters` – L1 ratio 0.5 and 10 cross-validation folds, with all other ElasticNetCV parameters (eps, number of alphas, max iterations, tolerance, etc.) at their scikit-learn defaults.
- `Manual parameters` – exposes the full parameter set: L1 ratio, cross-validation folds, `eps`, number of alphas, fit intercept, precompute strategy, max iterations, tolerance, copy X, verbosity, `n_jobs`, positive-coefficient constraint, `random_state`, and coefficient-update selection (cyclic/random).

### Rolling Window
This section configures how the rolling training/testing windows are generated:

- `Training Days` – one or more training-window lengths (in days) to evaluate. If multiple values are selected, the pipeline runs the full rolling-window evaluation once per training length. Generally with an increased lenght of training windows, the accuracy can be expected to improve, while the computational costs increase. Within the dataset used for the FOCCSI project, a 90 days training windows was found to be a good trade of the between accuracy and comuputational costs. The user may run an sensitivity analyis with this parameter and the given dataset

- `Forecast Horizon (days)` –  Defines the length of the test window that follows each training window. The standard value is one day. This reflects the requirements of the European electricity market, where the planned energy production for the following day must be submitted to the Transmission System Operator (TSO).

### Data Preprocessing
This section configures optional filtering steps applied before the combination models are trained:

**Filter benchmark data**\
When enabled, benchmark values below a configurable threshold are set to zero, in order to remove implausible or negligible benchmark readings before training.

**Filter forecast data**\
When enabled, threshold-based data-quality criteria are applied separately to the forecast horizon and historical forecast data. These include impossible values, missing values (NaNs), and consecutive repetitions of zero or non-zero values. If any of the specified thresholds is exceeded, the corresponding forecast feature is excluded from the forecast combination for that test day.

The following standard values are specific to the dataset used in this study and are based on a training window of 165 days; for smaller training windows, these values may need to be reduced accordingly. In general, the appropriate threshold values depend on the characteristics and availability of the underlying dataset.

| Criterion | Forecast Horizon – Wind | Forecast Horizon – PV | Historical – Wind | Historical – PV |
|---|---:|---:|---:|---:|
| Impossible values | 5 | 5 | 100 | 100 |
| Repeated zeros | 10 | 10 | 960 | 960 |
| Repeated non-zero values | 68 | 10 | 960 | 960 |
| Total missing values | 10 | 10 | 960 | 960 |
| Consecutive missing values | 5 | 5 | 96 | 96 |

Impossible values are defined as values below 0 or above the installed capacity.

The main difference between Wind and PV is the allowed number of consecutive repeated non-zero values in the forecast horizon: 68 for Wind versus 10 for PV, reflecting the different characteristics of the two generation technologies.

The user migh

## Run
The Run tab is used to finalize the output settings, optionally restrict the pipeline to a specific test period, and start (or abort) the FOCCSI pipeline execution.

### Output Settings
`Output path` specifies where result files are written. Relative paths are resolved from the project root.

### Testing
The `Test data selection` dropdown controls which time period the rolling-window evaluation covers:

- `Use all provided data` – the pipeline evaluates the full available time range of the merged input data.
- `Select test dates in range` – opens a popover with a date-range picker, letting the user restrict the evaluation to a specific first and last test date. Starting the run without selecting a valid range is blocked with a message.

### Starting and Aborting a Run
Clicking `START RUN` first analyzes the selected configuration:

- If required inputs/parameters are missing, or normalization is enabled without an uploaded installed-capacity file, a message describing the issue is shown and the run is not started.
- Otherwise, the FOCCSI pipeline is launched in the background, so the interface remains responsive while the run is in progress.

While a run is active, the tab shows a live progress bar and status text (current rolling-window step and test date), and an `Abort Run` button. Aborting stops the pipeline at the next safe checkpoint rather than instantly, since an in-progress model fit/window cannot be interrupted mid-computation. Once the run finishes (successfully, with an error, or aborted), the outcome is displayed and the `START RUN` button becomes available again.


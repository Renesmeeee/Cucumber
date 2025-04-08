# Cucumber Growth Model

A process-based simulation model for cucumber plant growth, focusing on canopy-level photosynthesis, biomass accumulation, and partitioning influenced by environmental conditions and management practices. Includes an optional dashboard for results visualization.

## Model Overview

This model simulates cucumber growth dynamically over time. Key aspects include:

*   **Process-Based:** Simulates core physiological and morphological processes like photosynthesis, respiration, biomass partitioning between organs (leaves, stem, fruit), and leaf area development.
*   **Canopy Focus:** Likely calculates photosynthesis based on canopy light interception and environmental factors affecting the entire plant stand.
*   **Environmentally Driven:** Growth is driven by environmental inputs like temperature, light (PPF/PAR), CO2 concentration, and humidity.
*   **Management Interaction:** Incorporates management practices such as leaf removal, which directly impacts canopy structure and resource allocation.
*   **Modular Structure:** Code is organized into modules within the `src/` directory, separating concerns like system definition, biological processes, environmental handling, and configuration.

## Project Structure

```
.
├── README.md               # This file: Project description, setup, and usage guide
├── pyproject.toml          # Project metadata and dependencies (Poetry)
├── poetry.lock             # Poetry lock file for reproducible dependencies
├── requirements.txt        # Project dependencies (pip format)
├── main.py                 # Main script to configure and run simulations
├── src/                    # Core source code for the growth model
│   ├── __init__.py         # Makes src a Python package
│   ├── configs/            # Contains model parameters, constants, and default settings (e.g., params.py)
│   ├── system/             # Defines the core system components, like the plant canopy (e.g., canopy.py)
│   ├── processes/          # Implements biological processes (e.g., photosynthesis, growth, partitioning)
│   ├── environment/        # Handles reading and processing environmental input data
│   ├── management/         # Implements crop management actions (e.g., leaf removal logic)
│   └── utils/              # Utility functions used across the model
├── data/                   # Directory for storing input data files
├── results/                # Directory where simulation output files are saved
└── dashboard/              # Code for the optional visualization dashboard
```

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   Required Python packages: Primarily `pandas`, `numpy`, `openpyxl`. See `pyproject.toml` or `requirements.txt` for a complete list (dashboard may have additional dependencies like `streamlit` or `plotly`).

### Installation

1.  **Clone the repository:**
    ```bash
    # git clone https://github.com/yourusername/cucumber_growth_model.git
    # cd cucumber_growth_model
    ```
2.  **Install dependencies:**
    *   **Using pip (recommended if not using Poetry):**
        ```bash
        pip install -r requirements.txt
        ```
    *   **Using Poetry:**
        ```bash
        # poetry install
        ```

### Running the Simulation

Simulations are launched using `main.py`. Configuration can be done via command-line arguments, which override defaults potentially set in `src/configs/params.py`.

1.  **Prepare Input Data:** Place your environmental and management data files (typically Excel `.xlsx` or `.xls`) in the `data/` directory. See the "Input Files Required" section for details on format and content.
2.  **Run from Command Line:**
    ```bash
    python main.py --file-path data/your_environment_data.xlsx --sheet-name Sheet1 --output-dir results/sim_run_1 [other options...]
    ```
    *   Use `--help` to see all available command-line options:
        ```bash
        python main.py --help
        ```
    *   **Key Arguments:**
        *   `--file-path`, `--sheet-name`: Specify the environmental data file and sheet.
        *   `--leaf-removal-file`, `--leaf-removal-sheet`: Specify leaf removal data (if using 'file' mode).
        *   `--fruit-dw-file`, `--fruit-dw-sheet`: Specify observed fruit dry weight data (for comparison).
        *   `--output-dir`: Specify the directory to save results.
        *   `--start-date`, `--initial-nodes`, `--sla`, `--plant-density`: Core simulation parameters.
        *   `--leaf-removal-mode`: Choose how leaf removal is handled ('file', 'interval', 'threshold').
        *   `--storage-format`: Output format ('excel', 'csv', 'pickle').
        *   `--ci-mode`: Method for internal CO2 calculation ('stomata', 'transpiration').
3.  **Run Interactively:**
    ```bash
    python main.py -i
    # or
    python main.py --interactive
    ```
    This mode will prompt you to select input files, sheets, and confirm parameters.

## Input Files Required

Input data files should be placed in the `data/` directory. The model primarily expects Excel files (`.xlsx`, `.xls`), but check `main.py` or relevant functions in `src/environment` if CSV is supported.

1.  **Environmental Data (Required):** Time-series data, typically hourly.
    *   **File:** Specified via `--file-path`.
    *   **Sheet:** Specified via `--sheet-name`.
    *   **Columns (Exact names might be configurable, check `src/configs/params.py` or data loading functions):**
        *   `DateTime`: Timestamp (e.g., `YYYY-MM-DD HH:MM:SS` or similar pandas-readable format).
        *   `Air temperature`: Ambient air temperature (Unit: °C).
        *   `PPF` or `PAR`: Photosynthetic Photon Flux (Unit: µmol/m²/s) or Photosynthetically Active Radiation (Unit: W/m²). Ensure the model parameter settings match the unit used.
        *   `CO2`: Atmospheric CO2 concentration (Unit: ppm or µmol/mol).
        *   `hs` or `RH`: Relative Humidity (Unit: %).
        *   *Optional:* `Ci` (Initial internal CO2, may be used depending on `--ci-mode`).
        *   *Other environmental variables might be used depending on model specifics (e.g., wind speed, VPD).*

2.  **Leaf Removal Data (Optional, depends on `--leaf-removal-mode`):**
    *   **Mode 'file':** Requires a separate file.
        *   **File:** Specified via `--leaf-removal-file`.
        *   **Sheet:** Specified via `--leaf-removal-sheet`.
        *   **Columns:** Typically `Date` and `Time` (or `DateTime`) indicating when removal occurs. May include specifics like number or rank of leaves removed.
    *   **Mode 'interval' / 'threshold':** Configured via command-line arguments like `--leaf-removal-interval`, `--max-leaves`, etc. No separate file needed.

3.  **Observed Fruit Dry Weight Data (Optional):** Used for model comparison/validation.
    *   **File:** Specified via `--fruit-dw-file`.
    *   **Sheet:** Specified via `--fruit-dw-sheet`.
    *   **Columns:** Typically `Date` and `Fruit DW/m^2` (Harvested fruit dry weight per square meter, Unit: g/m²).

*Note: Always verify expected column names, units, and date formats by checking the model's configuration (`src/configs/params.py`) and data loading logic, or by running with sample data.* 

## Output

Simulation results are saved in the directory specified by `--output-dir` (defaults usually to `results/`). The format and structure can be controlled via command-line arguments:

*   **Format (`--storage-format`):**
    *   `excel`: Saves multiple dataframes as sheets in a single `.xlsx` file (default).
    *   `csv`: Saves each dataframe as a separate `.csv` file.
    *   `pickle`: Saves dataframes as Python pickle files.
*   **Compression (`--compress`):** Can compress output files (e.g., `.csv.gz`).
*   **Content Control:** Flags like `--no-hourly-data`, `--no-daily-data`, `--no-leaf-data`, `--no-summary` can exclude specific parts of the output.

**Typical Output Files/Sheets:**

*   **Main Results (Hourly/Daily):** Time-series of key simulation variables.
    *   *File/Sheet Name Example:* `Results_hourly.csv`, `Results_daily.xlsx` (sheet: `Daily_Summary`)
    *   *Variables:* `DateTime`, `LAI` (Leaf Area Index), `Total_Biomass`, `Leaf_Biomass`, `Stem_Biomass`, `Fruit_Biomass` (g/m²), `Photosynthesis_Rate` (e.g., g CO2/m²/hr), `Respiration`, `Transpiration`, `Harvested_Fruit_DW`.
*   **Leaf Information:** Detailed tracking of individual leaves (can be large).
    *   *File/Sheet Name Examples:* `Leaf_Info.csv`, `Removed_Leaf_Info.xlsx`, `Remaining_Leaves_Info.xlsx`
    *   *Variables:* Leaf rank/ID, emergence time, expansion dynamics (area), senescence time, contribution to photosynthesis, removal time (if applicable).
*   **Photosynthesis Details:** More granular photosynthesis data.
    *   *File/Sheet Name Examples:* `Rank_Photosynthesis_Rates_GROSS.csv`, `Total_Photosynthesis_GROSS.xlsx`
    *   *Variables:* Photosynthesis rates broken down by leaf layer or rank, hourly/daily aggregated gross/net photosynthesis.
*   **Summary File:** High-level summary of the simulation run and key outputs.
    *   *File/Sheet Name Example:* `Simulation_Summary.txt` or `Summary` sheet in Excel.

*Note: Exact file/sheet names and variable columns depend on the specific implementation and output options chosen.* 

## Dashboard

A visualization dashboard might be available in the `dashboard/` directory, likely built using libraries like **Streamlit** or **Plotly Dash**.

*   **Check for Instructions:** Look for a `README.md` or comments within the scripts inside `dashboard/` for specific setup and launch instructions.
*   **Potential Launch Command (Example using Streamlit):**
    ```bash
    streamlit run dashboard/app.py # Or the specific name of the main dashboard script
    ```
*   The dashboard typically loads simulation results from the `results/` directory to display graphs of biomass accumulation, LAI development, photosynthesis rates, etc.

## License

This project is likely licensed under the MIT License, but check for a `LICENSE` file in the repository root for definitive details. 
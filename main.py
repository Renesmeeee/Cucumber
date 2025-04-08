"""
Main script for running cucumber growth model simulation
"""

import os
import sys
import argparse
import logging
import pandas as pd
from src.system.canopy import simulate_photosynthesis
from src.configs.params import (
    DEFAULT_SETTINGS,
    TIME_UNITS,
    LEAF_REMOVAL,
    STORAGE_FORMATS,
    GOMPERTZ_PARAMETERS,
)

# *** ADD Basic Logging Configuration ***
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
# *** END Logging Configuration ***


def create_parser():
    """
    Create argument parser for command line options

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Cucumber Growth Model Simulation")

    # Input/output paths
    parser.add_argument("--file-path", type=str, help="Path to environmental data file")
    parser.add_argument("--sheet-name", type=str, help="Sheet name in environmental data file")
    parser.add_argument("--leaf-removal-file", type=str, help="Path to leaf removal dates file")
    parser.add_argument("--leaf-removal-sheet", type=str, help="Sheet name for leaf removal dates")
    parser.add_argument("--fruit-dw-file", type=str, help="Path to fruit dry weight data file")
    parser.add_argument(
        "--fruit-dw-sheet", type=str, default="2", help="Sheet name for fruit dry weight data"
    )
    parser.add_argument(
        "--output-file", type=str, help="Path to save output file (for backward compatibility)"
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save results")

    # Simulation parameters
    parser.add_argument("--start-date", type=str, help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--initial-nodes", type=int, help="Initial number of nodes")
    parser.add_argument(
        "--threshold-before", type=float, help="Thermal time threshold before split date"
    )
    parser.add_argument(
        "--threshold-after", type=float, help="Thermal time threshold after split date"
    )
    parser.add_argument("--split-date", type=str, help="Date when threshold changes (YYYY-MM-DD)")
    parser.add_argument(
        "--target-fruit-weight",
        type=float,
        help="Target dry weight per fruit for harvest (g)",
    )
    parser.add_argument("--sla", type=float, help="Specific Leaf Area (m^2 g^-1)")
    parser.add_argument("--plant-density", type=float, help="Plant density per m^2")
    parser.add_argument(
        "--time-unit", type=str, choices=["minute", "hour"], help="Time unit for calculation"
    )

    # Leaf removal options
    parser.add_argument(
        "--leaf-removal-mode",
        type=str,
        choices=["file", "interval", "threshold"],
        help="Mode for leaf removal",
    )
    parser.add_argument("--leaf-removal-interval", type=int, help="Days between leaf removals")
    parser.add_argument(
        "--leaf-removal-time", type=str, help="Time of day for leaf removal (HH:MM:SS)"
    )
    parser.add_argument(
        "--leaf-removal-first-date",
        type=str,
        help="First date for interval-based removal (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--leaf-removal-first-leaf-count",
        type=int,
        help="First leaf count to trigger removal (interval mode)",
    )
    parser.add_argument(
        "--max-leaves", type=int, help="Maximum leaves before removal (threshold mode)"
    )
    parser.add_argument(
        "--leaves-to-remove", type=int, help="Number of leaves to remove (threshold mode)"
    )

    # Storage options
    parser.add_argument(
        "--storage-format",
        type=str,
        choices=list(STORAGE_FORMATS.keys()),
        help="Format for storing results",
    )
    parser.add_argument("--no-hourly-data", action="store_true", help="Do not save hourly data")
    parser.add_argument("--no-daily-data", action="store_true", help="Do not save daily data")
    parser.add_argument("--no-leaf-data", action="store_true", help="Do not save leaf-related data")
    parser.add_argument("--compress", action="store_true", help="Compress output files")
    parser.add_argument("--no-summary", action="store_true", help="Do not create summary file")
    parser.add_argument(
        "--split-hourly-files", action="store_true", help="Split hourly data into separate files"
    )

    # Interactive mode
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    # Add argument for Ci calculation mode
    parser.add_argument(
        "--ci-mode",
        type=str,
        choices=["stomata", "transpiration"],
        default="stomata",
        help="Method for calculating intercellular CO2 (Ci): 'stomata' (default) or 'transpiration' based.",
    )

    return parser


def list_excel_sheets(file_path):
    """
    List all sheets in an Excel file

    Args:
        file_path (str): Path to Excel file

    Returns:
        list: List of sheet names
    """
    try:
        xl = pd.ExcelFile(file_path)
        return xl.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []


def prompt_file_selection(message, default_path, file_type=None):
    """
    Prompt user to select a file

    Args:
        message (str): Message to display
        default_path (str): Default file path
        file_type (str): File type description

    Returns:
        str: Selected file path
    """
    print(f"\n{message}")
    print(f"  Default: {default_path or 'None'}")

    # Check if default file exists
    if default_path and os.path.isfile(default_path):
        print(f"  Default file exists: {default_path}")
    elif default_path:
        print(f"  Warning: Default file does not exist: {default_path}")

    # List files in current directory
    files = [f for f in os.listdir() if os.path.isfile(f)]
    if file_type:
        files = [f for f in files if f.endswith(file_type)]

    if files:
        print("  Available files:")
        for idx, file in enumerate(files, 1):
            print(f"    {idx}. {file}")

        choice = input(f"  Enter file number, path, or press Enter for default: ")

        if not choice:
            return default_path

        try:
            # If user entered a number, get the corresponding file
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        except ValueError:
            # User entered a path
            if os.path.isfile(choice):
                return choice
            elif os.path.isfile(os.path.join(os.getcwd(), choice)):
                return os.path.join(os.getcwd(), choice)
            else:
                print(f"  Warning: File not found: {choice}")
                if default_path and os.path.isfile(default_path):
                    print(f"  Using default: {default_path}")
                    return default_path
                else:
                    print("  No valid file selected.")
                    return None
    else:
        choice = input(f"  Enter file path or press Enter for default: ")
        if not choice:
            return default_path

        if os.path.isfile(choice):
            return choice
        elif os.path.isfile(os.path.join(os.getcwd(), choice)):
            return os.path.join(os.getcwd(), choice)
        else:
            print(f"  Warning: File not found: {choice}")
            if default_path and os.path.isfile(default_path):
                print(f"  Using default: {default_path}")
                return default_path
            else:
                print("  No valid file selected.")
                return None


def prompt_sheet_selection(file_path, default_sheet):
    """
    Prompt user to select a sheet from an Excel file

    Args:
        file_path (str): Path to Excel file
        default_sheet (str): Default sheet name

    Returns:
        str: Selected sheet name
    """
    if not file_path or not os.path.isfile(file_path):
        print("  No valid Excel file to select sheets from.")
        return default_sheet

    sheets = list_excel_sheets(file_path)

    if not sheets:
        print("  No sheets found in Excel file.")
        return default_sheet

    print(f"\nSelect sheet from {file_path}:")
    print(f"  Default: {default_sheet or 'None'}")
    print("  Available sheets:")

    for idx, sheet in enumerate(sheets, 1):
        print(f"    {idx}. {sheet}")

    choice = input("  Enter sheet number, name, or press Enter for default: ")

    if not choice:
        # If default sheet exists in the file, use it
        if default_sheet in sheets:
            return default_sheet
        # Otherwise, use the first sheet
        elif sheets:
            print(f"  Default sheet not found. Using first sheet: {sheets[0]}")
            return sheets[0]
        else:
            return default_sheet

    try:
        # If user entered a number, get the corresponding sheet
        idx = int(choice)
        if 1 <= idx <= len(sheets):
            return sheets[idx - 1]
    except ValueError:
        # User entered a sheet name
        if choice in sheets:
            return choice
        else:
            print(f"  Sheet not found: {choice}")
            # If default sheet exists in the file, use it
            if default_sheet in sheets:
                print(f"  Using default: {default_sheet}")
                return default_sheet
            # Otherwise, use the first sheet
            elif sheets:
                print(f"  Using first sheet: {sheets[0]}")
                return sheets[0]
            else:
                return default_sheet


def prompt_choice(message, choices, default=None):
    """
    Prompt user to select from a list of choices

    Args:
        message (str): Message to display
        choices (list): List of choices
        default: Default choice

    Returns:
        Selected choice
    """
    print(f"\n{message}")
    if default is not None:
        print(f"  Default: {default}")

    for idx, choice in enumerate(choices, 1):
        print(f"    {idx}. {choice}")

    user_choice = input("  Enter number, value, or press Enter for default: ")

    if not user_choice and default is not None:
        return default

    try:
        idx = int(user_choice)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    except ValueError:
        if user_choice in choices:
            return user_choice
        elif default is not None:
            print(f"  Invalid choice. Using default: {default}")
            return default
        else:
            print(f"  Invalid choice: {user_choice}")
            return choices[0]


def run_interactive_mode():
    """Run the model in interactive mode with user prompts for all options"""

    # Get default settings
    file_path = DEFAULT_SETTINGS.get("file_path", "data/input/climate_data.xlsx")
    sheet_name = DEFAULT_SETTINGS.get("sheet_name", "Sheet1")
    leaf_removal_dates_file = DEFAULT_SETTINGS.get(
        "leaf_removal_dates_file", "data/input/leaf_removal_dates.xlsx"
    )
    leaf_removal_sheet_name = DEFAULT_SETTINGS.get("leaf_removal_sheet_name", "Sheet1")
    leaf_removal_mode = DEFAULT_SETTINGS.get("leaf_removal_mode", "file")
    time_unit = DEFAULT_SETTINGS.get("time_unit", "hour")
    storage_format = DEFAULT_SETTINGS.get("storage_format", "excel")
    output_dir = DEFAULT_SETTINGS.get("output_dir", "results")

    print("=== Cucumber Growth Model - Interactive Mode ===")

    # Select environmental data file
    file_path = prompt_file_selection("Select environmental data file:", file_path, ".xlsx")

    if not file_path:
        print("Error: No environmental data file selected. Exiting.")
        return

    # Select sheet for environmental data
    sheet_name = prompt_sheet_selection(file_path, sheet_name)

    # Select leaf removal mode
    leaf_removal_mode = prompt_choice(
        "Select leaf removal mode:", ["file", "interval", "threshold"], leaf_removal_mode
    )

    # Set up leaf removal parameters based on mode
    leaf_removal_params = {}
    if leaf_removal_mode == "file":
        leaf_removal_dates_file = prompt_file_selection(
            "Select leaf removal dates file:", leaf_removal_dates_file, ".xlsx"
        )
        if leaf_removal_dates_file:
            leaf_removal_sheet_name = prompt_sheet_selection(
                leaf_removal_dates_file, leaf_removal_sheet_name
            )
    elif leaf_removal_mode == "interval":
        leaf_removal_interval_days = input(
            f"\nEnter days between leaf removals [Default: {DEFAULT_SETTINGS.get('leaf_removal_interval_days', 7)}]: "
        )
        leaf_removal_interval_days = (
            int(leaf_removal_interval_days)
            if leaf_removal_interval_days.strip()
            else DEFAULT_SETTINGS.get("leaf_removal_interval_days", 7)
        )

        leaf_removal_time = input(
            f"\nEnter time of day for leaf removal (HH:MM:SS) [Default: {DEFAULT_SETTINGS.get('leaf_removal_time', '08:00:00')}]: "
        )
        leaf_removal_time = (
            leaf_removal_time.strip()
            if leaf_removal_time.strip()
            else DEFAULT_SETTINGS.get("leaf_removal_time", "08:00:00")
        )

        # Ask for trigger method: date-based or leaf-count-based
        trigger_method = prompt_choice(
            "Select first removal trigger method:", ["date", "leaf-count"], "date"
        )

        if trigger_method == "date":
            leaf_removal_first_date = input(
                f"\nEnter first date for removal (YYYY-MM-DD) [Default: None]: "
            )
            leaf_removal_first_date = (
                leaf_removal_first_date.strip() if leaf_removal_first_date.strip() else None
            )
            leaf_removal_params["leaf_removal_first_date"] = leaf_removal_first_date
            leaf_removal_params["leaf_removal_first_leaf_count"] = None
        else:
            leaf_removal_first_leaf_count = input(
                f"\nEnter leaf count to trigger first removal [Default: 20]: "
            )
            try:
                leaf_removal_first_leaf_count = (
                    int(leaf_removal_first_leaf_count)
                    if leaf_removal_first_leaf_count.strip()
                    else 20
                )
                leaf_removal_params["leaf_removal_first_leaf_count"] = leaf_removal_first_leaf_count
                leaf_removal_params["leaf_removal_first_date"] = None
            except ValueError:
                print("Invalid leaf count. Using default: 20")
                leaf_removal_params["leaf_removal_first_leaf_count"] = 20
                leaf_removal_params["leaf_removal_first_date"] = None

        leaf_removal_params["leaf_removal_interval_days"] = leaf_removal_interval_days
        leaf_removal_params["leaf_removal_time"] = leaf_removal_time
    elif leaf_removal_mode == "threshold":
        max_leaves = input(
            f"\nEnter maximum number of leaves before removal [Default: {LEAF_REMOVAL['threshold'].get('max_leaves', 20)}]: "
        )
        max_leaves = (
            int(max_leaves)
            if max_leaves.strip()
            else LEAF_REMOVAL["threshold"].get("max_leaves", 20)
        )

        leaves_to_remove = input(
            f"\nEnter number of leaves to remove when threshold is exceeded [Default: {LEAF_REMOVAL['threshold'].get('leaves_to_remove', 5)}]: "
        )
        leaves_to_remove = (
            int(leaves_to_remove)
            if leaves_to_remove.strip()
            else LEAF_REMOVAL["threshold"].get("leaves_to_remove", 5)
        )

        leaf_removal_time = input(
            f"\nEnter time of day for leaf removal check (HH:MM:SS) [Default: {DEFAULT_SETTINGS.get('leaf_removal_time', '08:00:00')}]: "
        )
        leaf_removal_time = (
            leaf_removal_time.strip()
            if leaf_removal_time.strip()
            else DEFAULT_SETTINGS.get("leaf_removal_time", "08:00:00")
        )

        leaf_removal_params = {
            "leaf_removal_time": leaf_removal_time,
            "threshold": {"max_leaves": max_leaves, "leaves_to_remove": leaves_to_remove},
        }

    # *** NEW: Prompt for Target Fruit Weight ***
    target_fruit_weight_default = DEFAULT_SETTINGS.get("target_fruit_weight_per_node", 150.0)
    target_fruit_weight_input = input(
        f"\nEnter target fruit weight for harvest (g) [Default: {target_fruit_weight_default}]: "
    )
    try:
        target_fruit_weight = (
            float(target_fruit_weight_input)
            if target_fruit_weight_input.strip()
            else target_fruit_weight_default
        )
    except ValueError:
        print(f"Invalid input. Using default target fruit weight: {target_fruit_weight_default}")
        target_fruit_weight = target_fruit_weight_default
    # *** END NEW ***

    # Select time unit for calculation
    time_unit = prompt_choice("Select time unit for calculation:", ["minute", "hour"], time_unit)

    # Select Ci calculation mode
    ci_mode = prompt_choice(
        "Select Ci calculation mode:", ["stomata", "transpiration"], default="stomata"
    )

    # Select storage format
    storage_format = prompt_choice(
        "Select storage format for results:", list(STORAGE_FORMATS.keys()), storage_format
    )

    # Select output directory
    output_dir_input = input(f"\nEnter output directory [Default: {output_dir}]: ")
    if output_dir_input.strip():
        output_dir = output_dir_input.strip()
    # If no input (or only whitespace), keep the default value of output_dir

    # Make sure output_dir is not empty
    if not output_dir:
        output_dir = "results"
        print(f"Using default output directory: {output_dir}")

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare parameters for the final call --- # *** REVISED ***
    final_params = {
        "file_path": file_path,
        "sheet_name": sheet_name,
        "time_unit": time_unit,
        "storage_format": storage_format,
        "output_dir": output_dir,
        "ci_mode": ci_mode,
        "target_fruit_weight": target_fruit_weight,  # Add collected target weight
        # Add other simulation params from DEFAULT_SETTINGS or potentially prompted values
        "start_date": DEFAULT_SETTINGS.get("start_date"),
        "initial_nodes": DEFAULT_SETTINGS.get("initial_nodes"),
        "threshold_before": DEFAULT_SETTINGS.get("threshold_before"),
        "threshold_after": DEFAULT_SETTINGS.get("threshold_after"),
        "split_date": DEFAULT_SETTINGS.get("split_date"),
        "SLA": DEFAULT_SETTINGS.get("SLA"),
        "fruit_dw_file_path": DEFAULT_SETTINGS.get("fruit_dw_file_path"),
        "fruit_dw_sheet_name": DEFAULT_SETTINGS.get("fruit_dw_sheet_name"),
        "partitioning_vegetative_before": DEFAULT_SETTINGS.get("partitioning_vegetative_before"),
        "partitioning_fruit_before": DEFAULT_SETTINGS.get("partitioning_fruit_before"),
        "initial_remaining_vegetative_dw": DEFAULT_SETTINGS.get("initial_remaining_vegetative_dw"),
        "plant_density_per_m2": DEFAULT_SETTINGS.get("plant_density_per_m2"),
        # Add storage flags (consider prompting user for these too if needed)
        "save_hourly_data": DEFAULT_SETTINGS.get("save_hourly_data", True),
        "save_daily_data": DEFAULT_SETTINGS.get("save_daily_data", True),
        "save_leaf_data": DEFAULT_SETTINGS.get("save_leaf_data", True),
        "save_fruit": DEFAULT_SETTINGS.get("save_fruit", True),  # Add save_fruit flag
        "compress_output": DEFAULT_SETTINGS.get("compress_output", False),
        "create_summary": DEFAULT_SETTINGS.get("create_summary", True),
        "split_hourly_files": DEFAULT_SETTINGS.get("split_hourly_files", False),
    }

    # Add leaf removal specific params
    final_params["leaf_removal_dates_file"] = leaf_removal_dates_file  # Might be None
    final_params["leaf_removal_sheet_name"] = leaf_removal_sheet_name  # Might be None
    final_params.update(leaf_removal_params)  # Add mode-specific params like interval_days etc.

    # Run the simulation using run_with_leaf_removal_mode
    print(f"\n--- Starting Simulation ({leaf_removal_mode} mode) ---")
    run_with_leaf_removal_mode(mode=leaf_removal_mode, **final_params)
    print("--- Interactive Mode Simulation Finished ---")
    return final_params  # Return collected params (optional)


def main():
    """Main function to run the program"""
    parser = create_parser()
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()  # Just call it, it runs the simulation internally
        # # Ensure interactive mode parameters are passed correctly, including ci_mode if added there
        # # Currently, interactive mode doesn't explicitly handle ci_mode yet.
        # We might need to add ci_mode selection to run_interactive_mode or assume default.
        # For now, let's focus on non-interactive mode first.
        # simulate_photosynthesis(**interactive_params) # Example call - needs adjustment
        pass  # Placeholder - Interactive mode needs update for ci_mode
    else:
        # Non-interactive mode
        if args.file_path is None or args.sheet_name is None:
            parser.error("--file-path and --sheet-name are required in non-interactive mode.")

        params = {
            "file_path": args.file_path,
            "sheet_name": args.sheet_name,
            "start_date": args.start_date,
            "initial_nodes": args.initial_nodes,
            "threshold_before": args.threshold_before,
            "threshold_after": args.threshold_after,
            "split_date": args.split_date,
            "leaf_removal_dates_file": args.leaf_removal_dates_file,
            "leaf_removal_sheet_name": args.leaf_removal_sheet_name,
            "output_file_path": args.output_file_path,
            "SLA": args.sla,
            "fruit_dw_file_path": args.fruit_dw_file_path,
            "fruit_dw_sheet_name": args.fruit_dw_sheet_name,
            "partitioning_vegetative_before": args.partitioning_vegetative_before,
            "partitioning_fruit_before": args.partitioning_fruit_before,
            "initial_remaining_vegetative_dw": args.initial_remaining_vegetative_dw,
            "plant_density_per_m2": args.plant_density_per_m2,
            "time_unit": args.time_unit,
            "leaf_removal_interval_days": args.leaf_removal_interval_days,
            "leaf_removal_time": args.leaf_removal_time,
            "leaf_removal_first_date": args.leaf_removal_first_date,
            "leaf_removal_first_leaf_count": args.leaf_removal_first_leaf_count,
            "storage_format": args.storage_format,
            "output_dir": args.output_dir,
            "save_hourly_data": args.save_hourly_data,
            "save_daily_data": args.save_daily_data,
            "save_leaf_data": args.save_leaf_data,
            "compress_output": args.compress_output,
            "create_summary": args.create_summary,
            "split_hourly_files": args.split_hourly_files,
            "gompertz_params": GOMPERTZ_PARAMETERS,
            "ci_mode": args.ci_mode,
            "target_fruit_weight": args.target_fruit_weight,
        }

        # Run simulation with specified leaf removal mode
        run_with_leaf_removal_mode(args.leaf_removal_mode, **params)


def run_with_leaf_removal_mode(mode, **kwargs):
    """
    Run simulation with a specific leaf removal mode

    Args:
        mode (str): Leaf removal mode ('file', 'interval', or 'threshold')
        **kwargs: Additional parameters for the simulation
    """
    # Default file paths
    file_path = kwargs.get(
        "file_path", DEFAULT_SETTINGS.get("file_path", "data/input/climate_data.xlsx")
    )
    sheet_name = kwargs.get("sheet_name", DEFAULT_SETTINGS.get("sheet_name", "Sheet1"))

    # Ensure environment file exists
    if not os.path.isfile(file_path):
        print(f"Error: Environmental data file not found: {file_path}")
        print("Please ensure the climate data file exists in the correct location.")
        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
                print(f"Created directory: {data_dir}")
                print(f"Please place your climate data in: {file_path}")
            except Exception as e:
                print(f"Failed to create directory: {e}")
        return

    # Initialize leaf removal variables with default values
    leaf_removal_dates_file = None
    leaf_removal_sheet_name = None
    leaf_removal_interval_days = DEFAULT_SETTINGS.get("leaf_removal_interval_days", 7)
    leaf_removal_time = DEFAULT_SETTINGS.get("leaf_removal_time", "08:00:00")
    leaf_removal_first_date = DEFAULT_SETTINGS.get("leaf_removal_first_date", None)
    leaf_removal_first_leaf_count = DEFAULT_SETTINGS.get("leaf_removal_first_leaf_count", None)
    max_leaves = LEAF_REMOVAL["threshold"].get("max_leaves", 20)
    leaves_to_remove = LEAF_REMOVAL["threshold"].get("leaves_to_remove", 5)

    # Set mode-specific parameters
    if mode == "file":
        leaf_removal_dates_file = kwargs.get(
            "leaf_removal_dates_file",
            DEFAULT_SETTINGS.get("leaf_removal_dates_file", "data/input/leaf_removal_dates.xlsx"),
        )
        leaf_removal_sheet_name = kwargs.get(
            "leaf_removal_sheet_name", DEFAULT_SETTINGS.get("leaf_removal_sheet_name", "Sheet1")
        )

        # Ensure leaf removal file exists
        if not os.path.isfile(leaf_removal_dates_file):
            print(f"Warning: Leaf removal file not found: {leaf_removal_dates_file}")
            print("Running without leaf removal data.")
            leaf_removal_dates_file = None
            leaf_removal_sheet_name = None
    elif mode == "interval":
        leaf_removal_interval_days = kwargs.get(
            "leaf_removal_interval_days", DEFAULT_SETTINGS.get("leaf_removal_interval_days", 7)
        )
        leaf_removal_first_date = kwargs.get(
            "leaf_removal_first_date", DEFAULT_SETTINGS.get("leaf_removal_first_date", None)
        )
        leaf_removal_first_leaf_count = kwargs.get(
            "leaf_removal_first_leaf_count",
            DEFAULT_SETTINGS.get("leaf_removal_first_leaf_count", None),
        )
        leaf_removal_time = kwargs.get(
            "leaf_removal_time", DEFAULT_SETTINGS.get("leaf_removal_time", "08:00:00")
        )
    elif mode == "threshold":
        max_leaves = kwargs.get("max_leaves", LEAF_REMOVAL["threshold"].get("max_leaves", 20))
        leaves_to_remove = kwargs.get(
            "leaves_to_remove", LEAF_REMOVAL["threshold"].get("leaves_to_remove", 5)
        )
        leaf_removal_time = kwargs.get(
            "leaf_removal_time", DEFAULT_SETTINGS.get("leaf_removal_time", "08:00:00")
        )
    else:
        print(f"Invalid mode: {mode}")
        return None

    # Set other parameters
    fruit_dw_file_path = kwargs.get(
        "fruit_dw_file_path",
        DEFAULT_SETTINGS.get("fruit_dw_file_path", "data/input/fruit_dw_data.xlsx"),
    )
    fruit_dw_sheet_name = kwargs.get(
        "fruit_dw_sheet_name", DEFAULT_SETTINGS.get("fruit_dw_sheet_name", "2")
    )
    time_unit = kwargs.get("time_unit", DEFAULT_SETTINGS.get("time_unit", "hour"))

    # Set storage parameters
    storage_format = kwargs.get("storage_format", DEFAULT_SETTINGS.get("storage_format", "excel"))
    output_dir = kwargs.get("output_dir", DEFAULT_SETTINGS.get("output_dir", "results"))

    # Make sure output_dir is not empty
    if not output_dir:
        output_dir = "results"
        print(f"Output directory was empty, using default: {output_dir}")

    save_hourly_data = kwargs.get(
        "save_hourly_data", DEFAULT_SETTINGS.get("save_hourly_data", True)
    )
    save_daily_data = kwargs.get("save_daily_data", DEFAULT_SETTINGS.get("save_daily_data", True))
    save_leaf_data = kwargs.get("save_leaf_data", DEFAULT_SETTINGS.get("save_leaf_data", True))
    compress_output = kwargs.get("compress_output", DEFAULT_SETTINGS.get("compress_output", False))
    create_summary = kwargs.get("create_summary", DEFAULT_SETTINGS.get("create_summary", True))
    split_hourly_files = kwargs.get(
        "split_hourly_files", DEFAULT_SETTINGS.get("split_hourly_files", False)
    )

    # Determine output file path based on mode
    if mode == "file":
        filename_suffix = "file_removal"
    elif mode == "interval":
        filename_suffix = f"interval_{leaf_removal_interval_days}days"
    elif mode == "threshold":
        filename_suffix = f"threshold_{max_leaves}max_{leaves_to_remove}remove"

    output_file_path = kwargs.get(
        "output_file_path", os.path.join(output_dir, f"results_{filename_suffix}.xlsx")
    )

    # Print simulation settings
    print(f"Running simulation with:")
    print(f"  - Environment file: {file_path} (Sheet: {sheet_name})")
    print(f"  - Leaf removal mode: {mode}")
    if mode == "file":
        if leaf_removal_dates_file:
            print(
                f"  - Leaf removal file: {leaf_removal_dates_file} (Sheet: {leaf_removal_sheet_name})"
            )
        else:
            print(f"  - No leaf removal data - leaves will grow without removal")
    elif mode == "interval":
        print(f"  - Leaf removal interval: Every {leaf_removal_interval_days} days")
        if leaf_removal_first_date:
            print(f"  - First removal date: {leaf_removal_first_date}")
        elif leaf_removal_first_leaf_count:
            print(f"  - First removal when leaf count reaches: {leaf_removal_first_leaf_count}")
        else:
            print(f"  - First removal after {leaf_removal_interval_days} days from start")
        print(f"  - Removal time: {leaf_removal_time}")
    elif mode == "threshold":
        print(
            f"  - Leaf removal threshold: When leaves exceed {max_leaves}, remove {leaves_to_remove} leaves"
        )
        print(f"  - Check time: {leaf_removal_time}")
    print(f"  - Time unit: {time_unit}")
    print(f"  - Storage format: {storage_format}")
    print(f"  - Output path: {output_file_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare parameters for simulate_photosynthesis ---
    # Get flowering/fruit related parameters from kwargs or defaults
    split_date_val = kwargs.get("split_date", DEFAULT_SETTINGS.get("split_date"))
    target_fruit_weight_val = kwargs.get(
        "target_fruit_weight", DEFAULT_SETTINGS.get("target_fruit_weight_per_node", 150.0)
    )

    # Create the params dictionary for flowering/fruit logic
    flowering_params = {
        "split_date": split_date_val,
        "target_fruit_weight_per_node": target_fruit_weight_val,
        # Add other relevant params if canopy.py needs them later
    }

    # Run simulation with specified parameters
    result = simulate_photosynthesis(
        file_path=file_path,
        sheet_name=sheet_name,
        start_date=kwargs.get("start_date", DEFAULT_SETTINGS.get("start_date")),
        initial_nodes=kwargs.get("initial_nodes", DEFAULT_SETTINGS.get("initial_nodes")),
        threshold_before=kwargs.get("threshold_before", DEFAULT_SETTINGS.get("threshold_before")),
        threshold_after=kwargs.get("threshold_after", DEFAULT_SETTINGS.get("threshold_after")),
        # Pass the dictionary here
        params=flowering_params,
        # Remove split_date from here
        leaf_removal_dates_file=leaf_removal_dates_file,
        leaf_removal_sheet_name=leaf_removal_sheet_name,
        SLA=kwargs.get("SLA", DEFAULT_SETTINGS.get("SLA")),
        fruit_dw_file_path=fruit_dw_file_path,
        fruit_dw_sheet_name=fruit_dw_sheet_name,
        partitioning_vegetative_before=kwargs.get(
            "partitioning_vegetative_before",
            DEFAULT_SETTINGS.get("partitioning_vegetative_before"),
        ),
        partitioning_fruit_before=kwargs.get(
            "partitioning_fruit_before", DEFAULT_SETTINGS.get("partitioning_fruit_before")
        ),
        initial_remaining_vegetative_dw=kwargs.get(
            "initial_remaining_vegetative_dw",
            DEFAULT_SETTINGS.get("initial_remaining_vegetative_dw"),
        ),
        plant_density_per_m2=kwargs.get(
            "plant_density_per_m2", DEFAULT_SETTINGS.get("plant_density_per_m2")
        ),
        time_unit=time_unit,
        leaf_removal_mode=mode,
        leaf_removal_interval_days=leaf_removal_interval_days,
        leaf_removal_time=leaf_removal_time,
        leaf_removal_first_date=leaf_removal_first_date,
        leaf_removal_first_leaf_count=leaf_removal_first_leaf_count,
        storage_format=storage_format,
        output_dir=output_dir,
        save_hourly=kwargs.get("save_hourly_data", True),
        save_daily=kwargs.get("save_daily_data", True),
        save_leaf=kwargs.get("save_leaf_data", True),
        save_fruit=kwargs.get("save_fruit", True),
        compress=kwargs.get("compress_output", False),
        create_summary=kwargs.get("create_summary", True),
        split_hourly=kwargs.get("split_hourly_files", False),
        gompertz_params=kwargs.get("gompertz_params", GOMPERTZ_PARAMETERS),
        ci_mode=kwargs.get("ci_mode", "stomata"),
    )

    print(f"Simulation with {mode} leaf removal mode completed successfully!")
    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Check if non-interactive mode is explicitly requested with a parameter
    has_specific_param = any(
        [
            args.file_path,
            args.sheet_name,
            args.leaf_removal_file,
            args.leaf_removal_sheet,
            args.leaf_removal_mode,
            args.output_file,
            args.output_dir,
            args.leaf_removal_interval,
            args.leaf_removal_time,
            args.leaf_removal_first_date,
            args.leaf_removal_first_leaf_count,
            args.max_leaves,
            args.leaves_to_remove,
        ]
    )

    # Run in interactive mode by default, unless specific parameters are provided
    if args.interactive or not has_specific_param:
        run_interactive_mode()
    else:
        main()

    # 예제 사용법 (필요한 경우 주석 해제):

    # 다양한 저장 형식으로 실행:
    # run_with_leaf_removal_mode('file', storage_format='excel')
    # run_with_leaf_removal_mode('file', storage_format='csv', split_hourly_files=True)
    # run_with_leaf_removal_mode('file', storage_format='hdf5', compress_output=True)

    # 다양한 적엽 모드로 실행:
    # run_with_leaf_removal_mode('interval', leaf_removal_interval_days=7, leaf_removal_first_date='2021-03-01')
    # run_with_leaf_removal_mode('threshold', max_leaves=20, leaves_to_remove=5)

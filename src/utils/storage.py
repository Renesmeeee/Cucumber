"""
Data storage utilities for saving simulation results
"""

import os
import pandas as pd
import numpy as np
import json
import time
import zipfile
import h5py
import tempfile
import shutil
from datetime import datetime
from src.configs.params import STORAGE_FORMATS, OUTPUT_CATEGORIES, TIME_UNITS


class DataStorage:
    """
    Class for handling data storage operations with various formats
    """

    def __init__(
        self,
        output_dir="results",
        storage_format="excel",
        save_hourly=True,
        save_daily=True,
        save_leaf=True,
        compress_output=False,
        create_summary=True,
        split_hourly_files=False,
    ):
        """
        Initialize the DataStorage

        Args:
            output_dir (str): Directory to save results
            storage_format (str): Storage format ('excel', 'csv', or 'hdf5')
            save_hourly (bool): Whether to save hourly data
            save_daily (bool): Whether to save daily data
            save_leaf (bool): Whether to save leaf-related data
            compress_output (bool): Whether to compress output files
            create_summary (bool): Whether to create a summary file
            split_hourly_files (bool): Whether to split hourly data into separate files
        """
        self.output_dir = output_dir
        self.storage_format = storage_format.lower()
        self.save_hourly = save_hourly
        self.save_daily = save_daily
        self.save_leaf = save_leaf
        self.compress_output = compress_output
        self.create_summary = create_summary
        self.split_hourly_files = split_hourly_files

        # Get extension and whether this format uses a single file
        format_info = STORAGE_FORMATS.get(storage_format, STORAGE_FORMATS["excel"])
        self.extension = format_info["extension"]
        self.single_file = format_info["single_file"]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize metadata dictionary
        self.metadata = {
            "simulation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "storage_format": storage_format,
            "data_categories": {
                "hourly": save_hourly,
                "daily": save_daily,
                "leaf": save_leaf,
                "summary": create_summary,
            },
        }

    def generate_file_path(self, base_name=None, category=None, subcategory=None):
        """
        Generate file path based on storage format and settings

        Args:
            base_name (str): Base name for the file
            category (str): Data category ('hourly', 'daily', 'leaf', or 'summary')
            subcategory (str): Data subcategory (e.g., 'Results', 'Leaf_Info')

        Returns:
            str: Generated file path
        """
        if base_name is None:
            base_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.single_file:
            # For single file formats (Excel, HDF5)
            if category:
                file_name = f"{base_name}_{category}{self.extension}"
            else:
                file_name = f"{base_name}{self.extension}"
        else:
            # For multi-file formats (CSV)
            if subcategory:
                file_name = f"{base_name}_{subcategory}{self.extension}"
            elif category:
                file_name = f"{base_name}_{category}{self.extension}"
            else:
                file_name = f"{base_name}{self.extension}"

        return os.path.join(self.output_dir, file_name)

    def save_data(self, data_dict, base_name=None, simulation_params=None):
        """
        Save all simulation data

        Args:
            data_dict (dict): Dictionary containing all data frames
            base_name (str): Base name for the output files
            simulation_params (dict): Simulation parameters to include in metadata

        Returns:
            dict: Paths to saved files
        """
        # Update metadata with simulation parameters
        if simulation_params:
            self.metadata["simulation_parameters"] = simulation_params

        # Generate base name if not provided
        if base_name is None:
            base_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize result dictionary to store output file paths
        result_files = {}

        # Save data by format
        if self.storage_format == "excel":
            result_files = self._save_to_excel(data_dict, base_name)
        elif self.storage_format == "csv":
            result_files = self._save_to_csv(data_dict, base_name)
        elif self.storage_format == "hdf5":
            result_files = self._save_to_hdf5(data_dict, base_name)
        else:
            print(f"Unknown storage format: {self.storage_format}, defaulting to Excel")
            result_files = self._save_to_excel(data_dict, base_name)

        # Save metadata
        metadata_path = os.path.join(self.output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        result_files["metadata"] = metadata_path

        # Compress output if requested
        if self.compress_output:
            zip_path = os.path.join(self.output_dir, f"{base_name}_all_files.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for category, file_path in result_files.items():
                    if isinstance(file_path, dict):
                        for subcat, subpath in file_path.items():
                            if os.path.exists(subpath):
                                zipf.write(subpath, os.path.basename(subpath))
                    else:
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))

            result_files["compressed"] = zip_path

        return result_files

    def _save_to_excel(self, data_dict, base_name):
        """
        Save data to Excel format

        Args:
            data_dict (dict): Dictionary containing all data frames
            base_name (str): Base name for the output files

        Returns:
            dict: Paths to saved Excel files
        """
        result_files = {}

        # Create separate Excel files for hourly, daily, and leaf data
        if self.save_hourly:
            hourly_path = self.generate_file_path(base_name, "hourly")
            with pd.ExcelWriter(hourly_path) as writer:
                for sheet_name in OUTPUT_CATEGORIES["hourly"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        data_dict[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            result_files["hourly"] = hourly_path

        if self.save_daily:
            daily_path = self.generate_file_path(base_name, "daily")
            with pd.ExcelWriter(daily_path) as writer:
                for sheet_name in OUTPUT_CATEGORIES["daily"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        data_dict[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            result_files["daily"] = daily_path

        if self.save_leaf:
            leaf_path = self.generate_file_path(base_name, "leaf")
            with pd.ExcelWriter(leaf_path) as writer:
                for sheet_name in OUTPUT_CATEGORIES["leaf"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        data_dict[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            result_files["leaf"] = leaf_path

        # Create a summary file if requested
        if self.create_summary:
            # Create summary data - extract key statistics from daily data
            summary_data = self._create_summary_data(data_dict)
            summary_path = self.generate_file_path(base_name, "summary")

            with pd.ExcelWriter(summary_path) as writer:
                for sheet_name, df in summary_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            result_files["summary"] = summary_path

        # Create a combined file with all data
        all_data_path = self.generate_file_path(base_name, "all_data")
        with pd.ExcelWriter(all_data_path) as writer:
            for category in ["hourly", "daily", "leaf"]:
                if getattr(self, f"save_{category}"):
                    for sheet_name in OUTPUT_CATEGORIES[category]:
                        if sheet_name in data_dict and not data_dict[sheet_name].empty:
                            data_dict[sheet_name].to_excel(
                                writer, sheet_name=sheet_name, index=False
                            )

        result_files["all_data"] = all_data_path

        return result_files

    def _save_to_csv(self, data_dict, base_name):
        """
        Save data to CSV format (multiple files)

        Args:
            data_dict (dict): Dictionary containing all data frames
            base_name (str): Base name for the output files

        Returns:
            dict: Paths to saved CSV files
        """
        result_files = {}

        # Create a subdirectory for this simulation
        sim_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(sim_dir, exist_ok=True)

        # Save hourly data
        if self.save_hourly:
            hourly_files = {}
            hourly_dir = os.path.join(sim_dir, "hourly")
            os.makedirs(hourly_dir, exist_ok=True)

            for sheet_name in OUTPUT_CATEGORIES["hourly"]:
                if sheet_name in data_dict and not data_dict[sheet_name].empty:
                    if self.split_hourly_files and sheet_name == "Results":
                        # Split the hourly data by dates (e.g., by month)
                        hourly_files.update(
                            self._split_and_save_csv(data_dict[sheet_name], sheet_name, hourly_dir)
                        )
                    else:
                        file_path = os.path.join(hourly_dir, f"{sheet_name}.csv")
                        data_dict[sheet_name].to_csv(file_path, index=False)
                        hourly_files[sheet_name] = file_path

            result_files["hourly"] = hourly_files

        # Save daily data
        if self.save_daily:
            daily_files = {}
            daily_dir = os.path.join(sim_dir, "daily")
            os.makedirs(daily_dir, exist_ok=True)

            for sheet_name in OUTPUT_CATEGORIES["daily"]:
                if sheet_name in data_dict and not data_dict[sheet_name].empty:
                    file_path = os.path.join(daily_dir, f"{sheet_name}.csv")
                    data_dict[sheet_name].to_csv(file_path, index=False)
                    daily_files[sheet_name] = file_path

            result_files["daily"] = daily_files

        # Save leaf data
        if self.save_leaf:
            leaf_files = {}
            leaf_dir = os.path.join(sim_dir, "leaf")
            os.makedirs(leaf_dir, exist_ok=True)

            for sheet_name in OUTPUT_CATEGORIES["leaf"]:
                if sheet_name in data_dict and not data_dict[sheet_name].empty:
                    file_path = os.path.join(leaf_dir, f"{sheet_name}.csv")
                    data_dict[sheet_name].to_csv(file_path, index=False)
                    leaf_files[sheet_name] = file_path

            result_files["leaf"] = leaf_files

        # Create summary data
        if self.create_summary:
            summary_data = self._create_summary_data(data_dict)
            summary_files = {}
            summary_dir = os.path.join(sim_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)

            for name, df in summary_data.items():
                file_path = os.path.join(summary_dir, f"{name}.csv")
                df.to_csv(file_path, index=False)
                summary_files[name] = file_path

            result_files["summary"] = summary_files

        return result_files

    def _save_to_hdf5(self, data_dict, base_name):
        """
        Save data to HDF5 format

        Args:
            data_dict (dict): Dictionary containing all data frames
            base_name (str): Base name for the output files

        Returns:
            dict: Paths to saved HDF5 files
        """
        result_files = {}
        hdf5_path = self.generate_file_path(base_name)

        # Save data to HDF5 file
        with pd.HDFStore(hdf5_path, mode="w") as store:
            # Save hourly data
            if self.save_hourly:
                for sheet_name in OUTPUT_CATEGORIES["hourly"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        store.put(f"hourly/{sheet_name}", data_dict[sheet_name], format="table")

            # Save daily data
            if self.save_daily:
                for sheet_name in OUTPUT_CATEGORIES["daily"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        store.put(f"daily/{sheet_name}", data_dict[sheet_name], format="table")

            # Save leaf data
            if self.save_leaf:
                for sheet_name in OUTPUT_CATEGORIES["leaf"]:
                    if sheet_name in data_dict and not data_dict[sheet_name].empty:
                        store.put(f"leaf/{sheet_name}", data_dict[sheet_name], format="table")

            # Create and save summary data
            if self.create_summary:
                summary_data = self._create_summary_data(data_dict)
                for name, df in summary_data.items():
                    store.put(f"summary/{name}", df, format="table")

        result_files["hdf5"] = hdf5_path
        return result_files

    def _split_and_save_csv(self, df, name, directory):
        """
        Split large dataframe by date and save to separate CSV files

        Args:
            df (DataFrame): DataFrame to split and save
            name (str): Base name for the files
            directory (str): Directory to save files

        Returns:
            dict: Paths to saved CSV files
        """
        result_files = {}

        # Check if dataframe has a date/time column
        if "DateTime" in df.columns:
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            # Group by month
            grouped = df.groupby(pd.Grouper(key="DateTime", freq="M"))

            for name, group in grouped:
                month_str = name.strftime("%Y_%m")
                file_path = os.path.join(directory, f"{name}_{month_str}.csv")
                group.to_csv(file_path, index=False)
                result_files[f"{name}_{month_str}"] = file_path
        else:
            # If no date column, split by number of rows
            chunk_size = 10000  # Adjust as needed
            for i, chunk in enumerate(np.array_split(df, max(1, len(df) // chunk_size))):
                file_path = os.path.join(directory, f"{name}_part{i+1}.csv")
                pd.DataFrame(chunk).to_csv(file_path, index=False)
                result_files[f"{name}_part{i+1}"] = file_path

        return result_files

    def _create_summary_data(self, data_dict):
        """
        Create summary data from simulation results

        Args:
            data_dict (dict): Dictionary containing all data frames

        Returns:
            dict: Dictionary of summary DataFrames
        """
        summary_data = {}

        # Extract daily totals
        if "Daily_Photosynthesis_GROSS" in data_dict:
            daily_df = data_dict["Daily_Photosynthesis_GROSS"]

            if not daily_df.empty:
                # Create overall simulation summary
                overall_stats = {
                    "Metric": [
                        "Total Growth Period (days)",
                        "Avg Daily Photosynthesis",
                        "Max Daily Photosynthesis",
                        "Total Vegetative DW Production",
                        "Total Fruit DW Production",
                        "Total Leaf Area Removed",
                        "Total Fruit Harvested",
                    ],
                    "Value": [
                        len(daily_df),
                        daily_df["A_gross CH2O PRODUCTION (g/m^2/day)"].mean(),
                        daily_df["A_gross CH2O PRODUCTION (g/m^2/day)"].max(),
                        daily_df["Estimated Vegetative DW Production (g/m^2/d)"].sum(),
                        daily_df["Estimated Fruit DW Production (g/m^2/d)"].sum(),
                        daily_df["Removed_Leaf_Area"].sum(),
                        (
                            daily_df["Harvested Fruit DW (g/m^2)"].sum()
                            if "Harvested Fruit DW (g/m^2)" in daily_df.columns
                            else 0
                        ),
                    ],
                }

                summary_data["Overall_Summary"] = pd.DataFrame(overall_stats)

                # Create monthly summary
                if "Date" in daily_df.columns:
                    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
                    monthly_df = (
                        daily_df.groupby(pd.Grouper(key="Date", freq="M"))
                        .agg(
                            {
                                "A_gross CH2O PRODUCTION (g/m^2/day)": "sum",
                                "Estimated Vegetative DW Production (g/m^2/d)": "sum",
                                "Estimated Fruit DW Production (g/m^2/d)": "sum",
                                "Removed_Leaf_Area": "sum",
                                "Harvested Fruit DW (g/m^2)": (
                                    "sum"
                                    if "Harvested Fruit DW (g/m^2)" in daily_df.columns
                                    else "count"
                                ),
                            }
                        )
                        .reset_index()
                    )

                    monthly_df.rename(
                        columns={
                            "Date": "Month",
                            "A_gross CH2O PRODUCTION (g/m^2/day)": "Monthly CH2O Production",
                            "Estimated Vegetative DW Production (g/m^2/d)": "Monthly Vegetative DW",
                            "Estimated Fruit DW Production (g/m^2/d)": "Monthly Fruit DW",
                            "Removed_Leaf_Area": "Monthly Leaf Area Removed",
                            "Harvested Fruit DW (g/m^2)": "Monthly Fruit Harvested",
                        },
                        inplace=True,
                    )

                    summary_data["Monthly_Summary"] = monthly_df

        # Create leaf summary if available
        if "Leaf_Info" in data_dict and "Removed_Leaf_Info" in data_dict:
            leaf_df = data_dict["Leaf_Info"]
            removed_df = data_dict["Removed_Leaf_Info"]

            if not leaf_df.empty:
                leaf_stats = {
                    "Metric": [
                        "Total Leaves Generated",
                        "Total Leaves Removed",
                        "Remaining Leaves",
                        "Leaf Removal Events",
                        "Average Leaf Area (cm²)",
                    ],
                    "Value": [
                        len(leaf_df),
                        len(removed_df) if not removed_df.empty else 0,
                        len(leaf_df) - (len(removed_df) if not removed_df.empty else 0),
                        removed_df["Date"].nunique() if not removed_df.empty else 0,
                        leaf_df["Leaf Area"].mean() if "Leaf Area" in leaf_df.columns else 0,
                    ],
                }

                summary_data["Leaf_Summary"] = pd.DataFrame(leaf_stats)

        return summary_data


def save_simulation_results(
    result_data,
    output_dir="results",
    output_file=None,
    storage_format="excel",
    save_hourly=True,
    save_daily=True,
    save_leaf=True,
    save_fruit=True,
    compress_output=False,
    create_summary=True,
    split_hourly_files=False,
    time_unit="hour",
):
    """
    Save simulation results to specified format.

    Args:
        result_data (dict): Dictionary containing DataFrames of simulation results.
        output_dir (str): Directory to save the output files.
        output_file (str, optional): Specific output file path. Overrides default naming.
        storage_format (str): Format to save results ('excel', 'csv', 'hdf5').
        save_hourly (bool): Whether to save hourly data.
        save_daily (bool): Whether to save daily data.
        save_leaf (bool): Whether to save leaf-related data.
        save_fruit (bool): Whether to save fruit-related data.
        compress_output (bool): Whether to compress output files into a zip archive.
        create_summary (bool): Whether to create a summary file.
        split_hourly_files (bool): Whether to split large hourly files (e.g., for CSV).
        time_unit (str): Time unit used in the simulation ('hour' or 'minute').

    Returns:
        str: Path to the main saved file or directory.
    """
    start_time = time.time()

    # Determine base filename
    if output_file:
        output_dir = os.path.dirname(output_file) or output_dir
        base_filename = os.path.splitext(os.path.basename(output_file))[0]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"simulation_{timestamp}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to directory: {output_dir}")

    # Legacy Excel handling (if output_file is provided and ends with .xlsx)
    if output_file and storage_format == "excel":
        print("Using legacy Excel saving method due to specific output_file path.")
        try:
            save_to_excel_legacy(result_data, output_file, time_unit)
            end_time = time.time()
            print(
                f"Legacy Excel results saved to {output_file} (took {end_time - start_time:.2f}s)"
            )
            return output_file
        except Exception as e:
            print(f"Error saving with legacy Excel method: {e}")
            # Fallback to default method below if legacy fails
            print("Falling back to default saving method.")
            storage_format = "excel"  # Ensure format is set for fallback

    # Select saving function based on format
    if storage_format == "excel":
        saved_path = save_to_excel(
            result_data,
            output_dir,
            base_filename,
            save_hourly=save_hourly,
            save_daily=save_daily,
            save_leaf=save_leaf,
            save_fruit=save_fruit,
            split_hourly=split_hourly_files,
            time_unit=time_unit,
        )
    elif storage_format == "csv":
        saved_path = save_to_csv(
            result_data,
            output_dir,
            base_filename,
            save_hourly=save_hourly,
            save_daily=save_daily,
            save_leaf=save_leaf,
            save_fruit=save_fruit,
            split_hourly=split_hourly_files,
            time_unit=time_unit,
        )
    elif storage_format == "hdf5":
        saved_path = save_to_hdf5(
            result_data,
            output_dir,
            base_filename,
            save_hourly=save_hourly,
            save_daily=save_daily,
            save_leaf=save_leaf,
            save_fruit=save_fruit,
            time_unit=time_unit,
        )
    else:
        print(f"Unsupported storage format: {storage_format}. Defaulting to Excel.")
        saved_path = save_to_excel(
            result_data,
            output_dir,
            base_filename,
            save_hourly=save_hourly,
            save_daily=save_daily,
            save_leaf=save_leaf,
            save_fruit=save_fruit,
            split_hourly=split_hourly_files,
            time_unit=time_unit,
        )

    # Create summary file if requested
    summary_path = None
    if create_summary:
        try:
            summary_path = create_summary_file(result_data, output_dir, base_filename, time_unit)
            print(f"Summary file created at: {summary_path}")
        except Exception as e:
            print(f"Failed to create summary file: {e}")

    # Compress output if requested
    zip_path = None
    if compress_output:
        zip_path = os.path.join(output_dir, f"{base_filename}_results.zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add main saved file/directory
                if os.path.isfile(saved_path):
                    zipf.write(saved_path, os.path.basename(saved_path))
                elif os.path.isdir(saved_path):
                    for root, _, files in os.walk(saved_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
                # Add summary file if created
                if summary_path and os.path.exists(summary_path):
                    zipf.write(summary_path, os.path.basename(summary_path))
            print(f"Results compressed into: {zip_path}")
            # Optionally remove original files after compression
            # ... (add cleanup logic if needed) ...
        except Exception as e:
            print(f"Failed to compress output: {e}")

    end_time = time.time()
    print(f"Results saving completed in {end_time - start_time:.2f} seconds.")

    return zip_path if compress_output else saved_path


def save_to_excel_legacy(result_data, output_file_path, time_unit):
    """Legacy function to maintain compatibility with old Excel-only output"""
    time_label = TIME_UNITS[time_unit]["label"] if time_unit in TIME_UNITS else "hr"

    with pd.ExcelWriter(output_file_path) as writer:
        # Common data
        # Use .get() to avoid KeyError if 'Results' doesn't exist
        results_df = result_data.get("Results")
        if results_df is not None:
            results_df.reset_index().rename(columns={"index": "DateTime"})
            # Ensure DateTime is parsed correctly if it's not already datetime objects
            results_df["DateTime"] = pd.to_datetime(results_df["DateTime"])
            start_date = pd.to_datetime(results_df["DateTime"].min()).strftime("%Y-%m-%d")
            end_date = pd.to_datetime(results_df["DateTime"].max()).strftime("%Y-%m-%d")
            total_days = (results_df["DateTime"].max() - results_df["DateTime"].min()).days + 1

            results_df.to_excel(writer, sheet_name="Hourly_Results", index=False)

        # Leaf data
        leaf_info_df = result_data.get("Leaf_Info")
        if leaf_info_df is not None:
            leaf_info_df.to_excel(writer, sheet_name="Leaf_Info", index=False)

        removed_leaf_df = result_data.get("Removed_Leaf_Info")
        if removed_leaf_df is not None and not removed_leaf_df.empty:
            removed_leaf_df.to_excel(writer, sheet_name="Removed_Leaves", index=False)

        remaining_leaves_df = result_data.get("Remaining_Leaves_Info")
        if remaining_leaves_df is not None:
            remaining_leaves_df.to_excel(writer, sheet_name="Remaining_Leaves", index=False)

        # Photosynthesis data
        rank_photo_df = result_data.get("Rank_Photosynthesis_Rates_GROSS")
        if rank_photo_df is not None:
            rank_photo_df.to_excel(
                writer, sheet_name="Rank_Photo_Gross", index=False
            )  # Shortened sheet name

        total_photo_df = result_data.get("Total_Photosynthesis_GROSS")
        if total_photo_df is not None:
            total_photo_df.to_excel(
                writer, sheet_name="Total_Photo_Gross", index=False
            )  # Shortened sheet name

        # Use the correct key 'Daily_Calculation_Summary' and shorten sheet name
        daily_summary_df = result_data.get("Daily_Calculation_Summary")
        if daily_summary_df is not None:
            daily_summary_df.to_excel(writer, sheet_name="Daily_Summary", index=False)

        # Add calculation information sheet
        calculation_info = pd.DataFrame(
            [
                {"Parameter": "Time Unit", "Value": time_unit},
                {"Parameter": "Time Unit Label", "Value": time_label},
                {
                    "Parameter": "Seconds per Unit",
                    "Value": TIME_UNITS[time_unit]["seconds"] if time_unit in TIME_UNITS else 3600,
                },
                {"Parameter": "Storage Format", "Value": "excel"},
                {
                    "Parameter": "Creation Date",
                    "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            ]
        )
        calculation_info.to_excel(writer, sheet_name="Calc_Info", index=False)

    print(f"Results saved to {output_file_path}")


def save_to_excel(
    result_data,
    output_dir,
    base_filename,
    save_hourly=True,
    save_daily=True,
    save_leaf=True,
    save_fruit=True,
    split_hourly=False,
    time_unit="hour",
):
    """Saves simulation results into an Excel file.

    Args:
        result_data (dict): Dictionary of DataFrames.
        output_dir (str): Directory to save the file.
        base_filename (str): Base name for the file.
        save_hourly (bool): Save hourly data.
        save_daily (bool): Save daily data.
        save_leaf (bool): Save leaf data.
        save_fruit (bool): Save fruit data.
        split_hourly (bool): If True, attempts to split hourly sheets (less common for Excel).
        time_unit (str): Time unit label.

    Returns:
        str or None: Path to the saved Excel file, or None if saving failed or no data to save.
    """
    excel_file_path = os.path.join(output_dir, f"{base_filename}.xlsx")
    sheets_to_write = {}

    # Check which sheets have data and are requested to be saved
    if save_hourly:
        for key in OUTPUT_CATEGORIES.get("hourly", []):
            if key in result_data and not result_data[key].empty:
                sheets_to_write[key] = result_data[key]
    if save_daily:
        for key in OUTPUT_CATEGORIES.get("daily", []):
            if key in result_data and not result_data[key].empty:
                sheets_to_write[key] = result_data[key]
    if save_leaf:
        for key in OUTPUT_CATEGORIES.get("leaf", []):
            if key in result_data and not result_data[key].empty:
                sheets_to_write[key] = result_data[key]
    if save_fruit:
        for key in OUTPUT_CATEGORIES.get("fruit", []):
            if key in result_data and not result_data[key].empty:
                sheets_to_write[key] = result_data[key]

    # Only proceed if there is at least one sheet to write
    if not sheets_to_write:
        print(
            f"Warning: No data available to save in Excel file for {base_filename}. Skipping Excel creation."
        )
        return None

    print(
        f"Attempting to write the following sheets to {excel_file_path}: {list(sheets_to_write.keys())}"
    )

    try:
        with pd.ExcelWriter(excel_file_path) as writer:
            for sheet_name, df in sheets_to_write.items():
                print(
                    f"Writing sheet: {sheet_name} ({len(df) if isinstance(df, pd.DataFrame) else 'N/A'} rows)"
                )
                # Check if df is actually a DataFrame
                if not isinstance(df, pd.DataFrame):
                    print(
                        f"Warning: Skipping sheet {sheet_name} as it is not a DataFrame (type: {type(df)})."
                    )
                    continue
                try:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Successfully wrote sheet: {sheet_name}")
                except Exception as sheet_error:
                    print(f"Error writing sheet '{sheet_name}': {sheet_error}")
                    # Depending on the error, you might want to re-raise or just log
                    # Raising here will stop the process and trigger the outer except block
                    raise sheet_error

        print(f"Excel results saved successfully to: {excel_file_path}")
        return excel_file_path
    except Exception as e:
        print(
            f"Error saving to Excel (overall error, possibly during finalization or due to sheet error): {e}"
        )
        # Attempt to clean up potentially empty/corrupted file
        if os.path.exists(excel_file_path):
            try:
                os.remove(excel_file_path)
            except OSError as remove_error:
                print(
                    f"Warning: Could not remove potentially corrupted Excel file {excel_file_path}: {remove_error}"
                )
        return None


def save_to_csv(
    result_data,
    output_dir,
    base_filename,
    save_hourly=True,
    save_daily=True,
    save_leaf=True,
    save_fruit=True,
    split_hourly=False,
    time_unit="hour",
):
    """Saves simulation results into CSV files.

    Args:
        result_data (dict): Dictionary of DataFrames.
        output_dir (str): Directory to save the files.
        base_filename (str): Base name for the files/subdirectory.
        save_hourly (bool): Save hourly data.
        save_daily (bool): Save daily data.
        save_leaf (bool): Save leaf data.
        save_fruit (bool): Save fruit data.
        split_hourly (bool): Split hourly files (e.g., by date).
        time_unit (str): Time unit label.

    Returns:
        str: Path to the directory containing CSV files.
    """
    csv_dir = os.path.join(output_dir, base_filename + "_csv")
    os.makedirs(csv_dir, exist_ok=True)
    saved_files = []

    try:
        if save_hourly:
            hourly_dir = os.path.join(csv_dir, "hourly")
            os.makedirs(hourly_dir, exist_ok=True)
            for key in OUTPUT_CATEGORIES["hourly"]:
                if key in result_data and not result_data[key].empty:
                    if split_hourly:
                        # Implement splitting logic if needed, e.g., by date
                        # For simplicity, saving as one file for now
                        file_path = os.path.join(hourly_dir, f"{key}.csv")
                        result_data[key].to_csv(file_path, index=False)
                        saved_files.append(file_path)
                    else:
                        file_path = os.path.join(hourly_dir, f"{key}.csv")
                        result_data[key].to_csv(file_path, index=False)
                        saved_files.append(file_path)

        if save_daily:
            daily_dir = os.path.join(csv_dir, "daily")
            os.makedirs(daily_dir, exist_ok=True)
            for key in OUTPUT_CATEGORIES["daily"]:
                if key in result_data and not result_data[key].empty:
                    file_path = os.path.join(daily_dir, f"{key}.csv")
                    result_data[key].to_csv(file_path, index=False)
                    saved_files.append(file_path)

        if save_leaf:
            leaf_dir = os.path.join(csv_dir, "leaf")
            os.makedirs(leaf_dir, exist_ok=True)
            for key in OUTPUT_CATEGORIES["leaf"]:
                if key in result_data and not result_data[key].empty:
                    file_path = os.path.join(leaf_dir, f"{key}.csv")
                    result_data[key].to_csv(file_path, index=False)
                    saved_files.append(file_path)

        if save_fruit:
            fruit_dir = os.path.join(csv_dir, "fruit")
            os.makedirs(fruit_dir, exist_ok=True)
            for key in OUTPUT_CATEGORIES["fruit"]:
                if key in result_data and not result_data[key].empty:
                    file_path = os.path.join(fruit_dir, f"{key}.csv")
                    result_data[key].to_csv(file_path, index=False)
                    saved_files.append(file_path)

        print(f"CSV results saved to directory: {csv_dir}")
        return csv_dir
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return None


def save_to_hdf5(
    result_data,
    output_dir,
    base_filename,
    save_hourly=True,
    save_daily=True,
    save_leaf=True,
    save_fruit=True,
    time_unit="hour",
):
    """Saves simulation results into an HDF5 file.

    Args:
        result_data (dict): Dictionary of DataFrames.
        output_dir (str): Directory to save the file.
        base_filename (str): Base name for the file.
        save_hourly (bool): Save hourly data.
        save_daily (bool): Save daily data.
        save_leaf (bool): Save leaf data.
        save_fruit (bool): Save fruit data.
        time_unit (str): Time unit label.

    Returns:
        str: Path to the saved HDF5 file.
    """
    hdf5_file_path = os.path.join(output_dir, f"{base_filename}.h5")
    try:
        with pd.HDFStore(hdf5_file_path, mode="w") as store:
            if save_hourly:
                for key in OUTPUT_CATEGORIES["hourly"]:
                    if key in result_data and not result_data[key].empty:
                        store.put(f"hourly/{key}", result_data[key], format="table")
            if save_daily:
                for key in OUTPUT_CATEGORIES["daily"]:
                    if key in result_data and not result_data[key].empty:
                        store.put(f"daily/{key}", result_data[key], format="table")
            if save_leaf:
                for key in OUTPUT_CATEGORIES["leaf"]:
                    if key in result_data and not result_data[key].empty:
                        store.put(f"leaf/{key}", result_data[key], format="table")
            if save_fruit:
                for key in OUTPUT_CATEGORIES["fruit"]:
                    if key in result_data and not result_data[key].empty:
                        store.put(f"fruit/{key}", result_data[key], format="table")

        print(f"HDF5 results saved to: {hdf5_file_path}")
        return hdf5_file_path
    except Exception as e:
        print(f"Error saving to HDF5: {e}")
        # Clean up potentially corrupted HDF5 file
        if os.path.exists(hdf5_file_path):
            try:
                os.remove(hdf5_file_path)
            except OSError:
                pass
        return None


def create_summary_file(result_data, output_dir, base_filename, time_unit="hour"):
    """Create a summary file with key statistics"""
    # Extract key info from result data
    try:
        time_label = TIME_UNITS[time_unit]["label"] if time_unit in TIME_UNITS else "hr"

        # Use 'daily_summary' DataFrame
        daily_df = result_data.get("daily_summary")
        if daily_df is None or daily_df.empty:
            print("Warning: 'daily_summary' data is missing or empty. Cannot create full summary.")
            return None

        # Ensure 'Date' column exists and is datetime
        if "Date" not in daily_df.columns:
            print("Warning: 'Date' column missing in 'daily_summary'. Cannot determine time range.")
            start_date = "N/A"
            end_date = "N/A"
            total_days = 0
        else:
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            start_date = pd.to_datetime(daily_df["Date"].min()).strftime("%Y-%m-%d")
            end_date = pd.to_datetime(daily_df["Date"].max()).strftime("%Y-%m-%d")
            total_days = (daily_df["Date"].max() - daily_df["Date"].min()).days + 1

        # Get photosynthesis info using correct column names from 'daily_summary'
        # Use .get(column, 0) or check existence to handle missing columns gracefully
        ps_prod_col = "A_grossCH2O_PRODUCTION (g/m^2/day)"
        veg_dw_col = "Estimated Vegetative DW Production (g/m^2/d)"
        fruit_dw_prod_col = "Estimated Fruit DW Production (g/m^2/d)"
        harvested_dw_col = "Harvested Fruit DW (g/m^2)"  # Or the internally calculated one
        removed_area_col = "Removed_Leaf_Area"

        total_gross_ch2o = daily_df.get(ps_prod_col, pd.Series(0)).sum()
        max_daily_gross_ch2o = daily_df.get(ps_prod_col, pd.Series(0)).max()
        mean_daily_gross_ch2o = daily_df.get(ps_prod_col, pd.Series(0)).mean()
        total_veg_dw = daily_df.get(veg_dw_col, pd.Series(0)).sum()
        total_fruit_prod_dw = daily_df.get(fruit_dw_prod_col, pd.Series(0)).sum()
        total_harvested_dw = daily_df.get(harvested_dw_col, pd.Series(0)).sum()
        total_removed_area = daily_df.get(removed_area_col, pd.Series(0)).sum()

        # Get leaf info using correct dictionary keys
        leaves_info_df = result_data.get("leaves_info")
        removed_leaves_df = result_data.get("removed_leaves_info")
        # 'Remaining_Leaves_Info' doesn't exist, calculate from final states or leaves_info/removed_leaves
        # Example calculation: total - removed. Use Node_Final_States if available for more accuracy.
        node_final_states = result_data.get("Node_Final_States")
        remaining_leaves_count = 0
        if node_final_states is not None and not node_final_states.empty:
            # Count nodes that were leaves and not harvested/failed (adjust states as needed)
            remaining_leaf_states = [
                "appeared",
                "bud",
                "flowering",
                "set",
            ]  # Example states considered 'remaining leaves'
            remaining_leaves_count = node_final_states[
                node_final_states["state"].isin(remaining_leaf_states)
            ].shape[0]
        elif leaves_info_df is not None and removed_leaves_df is not None:
            remaining_leaves_count = len(leaves_info_df) - len(removed_leaves_df)

        total_leaves = len(leaves_info_df) if leaves_info_df is not None else 0
        removed_leaves = len(removed_leaves_df) if removed_leaves_df is not None else 0

        # Prepare summary data with updated keys and calculations
        summary = {
            "Simulation Summary": {
                "Start Date": start_date,
                "End Date": end_date,
                "Total Days": total_days,
                "Time Unit": time_unit,
                "Total Gross CH2O Production (g/m^2)": round(total_gross_ch2o, 2),
                "Average Daily Gross CH2O Production (g/m^2/day)": round(mean_daily_gross_ch2o, 2),
                "Maximum Daily Gross CH2O Production (g/m^2/day)": round(max_daily_gross_ch2o, 2),
                "Total Vegetative DW Production (g/m^2)": round(total_veg_dw, 2),
                "Total Fruit DW Production (g/m^2)": round(
                    total_fruit_prod_dw, 2
                ),  # Potential growth allocated
                "Total Fruit DW Harvested (g/m^2)": round(
                    total_harvested_dw, 2
                ),  # Actual harvested
                "Total Leaf Area Removed (m^2/m^2?)": round(total_removed_area, 3),  # Check units
                "Total Leaves Generated": total_leaves,
                "Leaves Removed": removed_leaves,
                "Remaining Leaves (approx)": remaining_leaves_count,  # Indicate calculation method if needed
            }
        }

        # Add daily photosynthesis statistics (using corrected column names)
        # Select only relevant numeric columns for describe()
        numeric_cols_to_describe = [
            ps_prod_col,
            veg_dw_col,
            fruit_dw_prod_col,
            harvested_dw_col,
            removed_area_col,
            "Total_Fruit_DW (g/m^2)",
            "Active_Fruit_Count",
            "SO_SI_Ratio",  # Add other relevant daily cols
        ]
        valid_cols = [col for col in numeric_cols_to_describe if col in daily_df.columns]
        if valid_cols:
            daily_stats = daily_df[valid_cols].describe().to_dict()
            summary["Daily Statistics"] = {
                col: {
                    stat: round(value, 3) if isinstance(value, (int, float)) else value
                    for stat, value in stats.items()
                }
                for col, stats in daily_stats.items()
            }
        else:
            summary["Daily Statistics"] = "No numeric columns found for description."

        # Create file path
        summary_path = os.path.join(output_dir, f"{base_filename}_summary.json")

        # Save as JSON
        with open(summary_path, "w") as f:
            json.dump(
                summary, f, indent=2, default=str
            )  # Use default=str for non-serializable types like numpy numbers

        print(f"Summary file created at: {summary_path}")
        return summary_path

    except KeyError as e:
        print(
            f"Error creating summary file: Missing key {e}. Check result_data structure and column names."
        )
        return None
    except Exception as e:
        print(f"Error creating summary file: {e}")
        # Log traceback for detailed debugging
        import traceback

        traceback.print_exc()
        return None

"""
Climate data management for simulation
"""

import pandas as pd
from src.management.leaf_removal import generate_leaf_removal_dates
import numpy as np
import os


class SimulationInitializer:
    """
    Initialize simulation based on climate data
    """

    def __init__(
        self,
        file_path,
        sheet_name,
        start_date,
        initial_nodes,
        threshold_before,
        threshold_after,
        split_date,
        leaf_removal_dates_file=None,
        leaf_removal_sheet_name=None,
        leaf_removal_mode="file",
        leaf_removal_interval_days=7,
        leaf_removal_time="08:00:00",
        leaf_removal_first_date=None,
        leaf_removal_first_leaf_count=None,
    ):
        """
        Initialize the SimulationInitializer class

        Args:
            file_path (str): Path to climate data file
            sheet_name (str): Sheet name in climate data file
            start_date (str): Start date for simulation (YYYY-MM-DD)
            initial_nodes (int): Initial number of nodes
            threshold_before (float): Thermal time threshold before split date
            threshold_after (float): Thermal time threshold after split date
            split_date (str): Date when threshold changes (YYYY-MM-DD)
            leaf_removal_dates_file (str): Path to leaf removal dates file
            leaf_removal_sheet_name (str): Sheet name in leaf removal dates file
            leaf_removal_mode (str): Mode for leaf removal ('file', 'interval', or 'threshold')
            leaf_removal_interval_days (int): Days between leaf removals (for interval mode)
            leaf_removal_time (str): Time of day for leaf removal (HH:MM:SS)
            leaf_removal_first_date (str): First date for interval-based removal (YYYY-MM-DD)
            leaf_removal_first_leaf_count (int): First leaf count to trigger removal (interval mode)
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.start_date = start_date
        self.initial_nodes = initial_nodes
        self.threshold_before = threshold_before
        self.threshold_after = threshold_after
        self.split_date = split_date
        self.leaf_removal_dates_file = leaf_removal_dates_file
        self.leaf_removal_sheet_name = leaf_removal_sheet_name
        self.leaf_removal_mode = leaf_removal_mode
        self.leaf_removal_interval_days = leaf_removal_interval_days
        self.leaf_removal_time = leaf_removal_time
        self.leaf_removal_first_date = leaf_removal_first_date
        self.leaf_removal_first_leaf_count = leaf_removal_first_leaf_count

    def initialize(self):
        """
        Initialize simulation - simplified to match original Cucumber.py code

        Returns:
            tuple: (df, df_hourly, daily_avg_temp, leaf_removal_dates)
        """
        try:
            # Simple data loading and processing like in the original code
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df.set_index("DateTime", inplace=True)

            # --- Add Robust Data Type Conversion ---
            numeric_cols = [
                "Temperature",
                "temperature",  # Handle both possible names
                "RH",
                "hs",  # Include hs just in case it's in the input
                "Radiation intensity (W/m2)",
                "PAR",
                "PPF",  # Handle both possible names
                "CO2",
            ]

            for col in numeric_cols:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # --- End Data Type Conversion ---

            # Convert column names if needed (AFTER converting types)
            if "Air temperature" in df.columns:
                df.rename(columns={"Air temperature": "temperature"}, inplace=True)
            # Rename PPF to PARi if PARi doesn't exist AND PPF exists
            if "PPF" in df.columns and "PARi" not in df.columns and "PAR" not in df.columns:
                df.rename(columns={"PPF": "PARi"}, inplace=True)
            # Rename PAR to PARi if PARi doesn't exist AND PAR exists
            elif "PAR" in df.columns and "PARi" not in df.columns:
                df.rename(columns={"PAR": "PARi"}, inplace=True)

            # Filter data after start date
            if self.start_date:
                df = df[df.index >= self.start_date]

            # Create hourly data
            hourly_climate_data = df.resample("h").mean(numeric_only=True)

            # Calculate daily average temperature
            daily_avg_temp = hourly_climate_data.resample("D").mean(numeric_only=True)
            if "temperature" in daily_avg_temp.columns:
                daily_avg_temp["daily_growing_temp"] = daily_avg_temp["temperature"].apply(
                    lambda x: x - 10 if x > 10 else 0
                )
            elif "Temperature" in daily_avg_temp.columns:
                daily_avg_temp.rename(columns={"Temperature": "temperature"}, inplace=True)
                daily_avg_temp["daily_growing_temp"] = daily_avg_temp["temperature"].apply(
                    lambda x: x - 10 if x > 10 else 0
                )

            # Convert index to date format
            daily_avg_temp.index = daily_avg_temp.index.date

            # Load leaf removal dates
            if (
                self.leaf_removal_mode == "file"
                and self.leaf_removal_dates_file
                and os.path.isfile(self.leaf_removal_dates_file)
            ):
                try:
                    leaf_removal_dates = pd.read_excel(
                        self.leaf_removal_dates_file, sheet_name=self.leaf_removal_sheet_name
                    )
                    leaf_removal_dates["Date"] = pd.to_datetime(leaf_removal_dates["Date"]).dt.date
                except Exception as e:
                    print(f"Error reading leaf removal file: {e}")
                    leaf_removal_dates = pd.DataFrame(columns=["Date", "Time"])
            else:
                # Create empty leaf removal dataframe
                leaf_removal_dates = pd.DataFrame(columns=["Date", "Time"])

            return df, hourly_climate_data, daily_avg_temp, leaf_removal_dates

        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise

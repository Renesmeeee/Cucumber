"""
Canopy system module for cucumber growth modeling
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import math
from src.environment.climate import SimulationInitializer
from src.processes.morphology.leaf_area import LeafCalculator
from src.processes.physiology.photosynthesis import (
    calculate_photosynthesis_dataframe,
    calculate_rm,
    FvCB_Calculator,
)  # Keep FvCB
from src.management.harvest import add_fruit_dw_column, process_harvest_data

# Removed MOL_MASS_CO2 as it's not directly used (cancels out in CH2O calc)
from src.configs.constants import GROWTH_EFFICIENCY, MOL_MASS_CH2O

# Removed DEFAULT_SETTINGS as it's not directly used in this function
from src.configs.params import TIME_UNITS, LEAF_REMOVAL
from src.utils.storage import save_simulation_results
from src.management.leaf_removal import process_leaf_removal  # Keep process_leaf_removal

# Import necessary functions from the updated flowering module
from src.processes.phenology.flowering import (
    calculate_flower_appearance_rate,
    calculate_potential_fruit_set,
    determine_actual_fruit_set,
    # interpolate_nr_params is used internally by calculate_potential_fruit_set
)

# Import constants and potentially parameters
from src.configs.constants import GROWTH_EFFICIENCY  # Assuming GROWTH_EFFICIENCY is here

# Assuming parameters like split_date, target_fruit_weight_per_node are loaded correctly
# For example:
# from src.configs.params import DEFAULT_SETTINGS
# split_date_str = DEFAULT_SETTINGS.get("split_date") # Get from loaded params/settings
# TARGET_FRUIT_WEIGHT_PER_NODE = DEFAULT_SETTINGS.get("target_fruit_weight_per_node", 150.0) # Example

# Ensure logger is initialized if not already present at module level
logger = logging.getLogger("canopy")


# --- Helper function for Vegetative Potential Growth ---
def calculate_potential_veg_growth(temperature: float) -> float:
    """
    Calculates potential vegetative growth rate based on temperature,
    using linear interpolation between calibrated points (18°C and 24°C).
    Clamps the rate outside this range based on the paper's description.

    Args:
        temperature: Average daily air temperature (°C).

    Returns:
        Potential vegetative growth rate (g/day). Based on whole plant? Assume g/plant/day for now.
        TODO: Confirm the unit basis (per plant or per m2) consistent with Sink demand. Assuming per plant for now matching fruit.
    """
    TEMP_VEG_1 = 18.0
    RATE_VEG_1 = 7.8
    TEMP_VEG_2 = 24.0
    RATE_VEG_2 = 9.3

    if temperature <= TEMP_VEG_1:
        return RATE_VEG_1
    elif temperature >= TEMP_VEG_2:
        return RATE_VEG_2
    else:
        # Linear interpolation
        rate = RATE_VEG_1 + (RATE_VEG_2 - RATE_VEG_1) * (temperature - TEMP_VEG_1) / (
            TEMP_VEG_2 - TEMP_VEG_1
        )
        return rate


# --- Helper function for Richards Function (Fruit Potential Growth) ---
# Constants and function based on Marcelis & Gijzen, 1998
M_FRUIT = 60.7  # g, Max potential fruit DW
B_FRUIT = 0.0170  # (°C·day)⁻¹
C_FRUIT = 131.0  # °C·day
D_FRUIT = 0.0111  # Dimensionless
BASE_TEMP_FRUIT = 10.0  # °C


def calculate_potential_fruit_growth(accum_degree_days: float, temperature: float) -> float:
    """
    Calculates potential fruit growth rate for a single fruit using Richards function.
    Formula from Marcelis PhD Thesis (Ch 5, Eq 6). Y_pot,i

    Args:
        accum_degree_days (X_i): Accumulated degree days (°C·day) since fruit set.
        temperature (T): Current average daily air temperature (°C).

    Returns:
        Potential fruit growth rate (g DM / fruit / day). Returns 0 if temp <= base_temp.
    """
    if temperature <= BASE_TEMP_FRUIT:
        return 0.0

    temp_effect = temperature - BASE_TEMP_FRUIT
    exp_term = math.exp(-B_FRUIT * (accum_degree_days - C_FRUIT))
    denominator_base = 1 + D_FRUIT * exp_term
    # Avoid division by zero or power of zero issues if denominator_base is <= 0
    if denominator_base <= 1e-9:
        logger.warning(
            f"Richards function denominator base near zero/negative ({denominator_base:.2e}) for X_i={accum_degree_days:.1f}. Growth set to 0."
        )
        return 0.0

    # Avoid issues with potentially very large exponents if 1/D_FRUIT is large
    # D_FRUIT is small, so 1+1/D_FRUIT is large. Let's compute safely.
    power_exponent = 1.0 + (1.0 / D_FRUIT) if D_FRUIT > 1e-9 else 1.0  # Handle D_FRUIT near zero

    try:
        denominator = denominator_base**power_exponent
        if denominator <= 1e-9:  # Check result of power too
            logger.warning(
                f"Richards function denominator near zero/negative after power for X_i={accum_degree_days:.1f}. Growth set to 0."
            )
            return 0.0

        potential_rate = temp_effect * (B_FRUIT * M_FRUIT * exp_term) / denominator

    except OverflowError:
        logger.warning(
            f"OverflowError in Richards function calculation for X_i={accum_degree_days:.1f}. Growth set to 0."
        )
        potential_rate = 0.0
    except ValueError:  # E.g., math domain error from power
        logger.warning(
            f"ValueError in Richards function calculation for X_i={accum_degree_days:.1f}. Growth set to 0."
        )
        potential_rate = 0.0

    # Potential rate should not be negative
    return max(0.0, potential_rate)


def simulate_photosynthesis(
    file_path,
    sheet_name,
    start_date=None,
    initial_nodes=5,
    threshold_before=21,
    threshold_after=33,
    leaf_removal_dates_file=None,
    leaf_removal_sheet_name=None,
    SLA=0.03,
    fruit_dw_file_path=None,
    fruit_dw_sheet_name="2",
    partitioning_vegetative_before=0.7,
    partitioning_fruit_before=0.3,
    initial_remaining_vegetative_dw=0.1,
    plant_density_per_m2=2.5,
    time_unit="hour",
    leaf_removal_mode="file",
    leaf_removal_interval_days=7,
    leaf_removal_time="08:00:00",
    leaf_removal_first_date=None,
    leaf_removal_first_leaf_count=None,
    storage_format="excel",
    output_dir=None,
    save_hourly=True,
    save_daily=True,
    save_leaf=True,
    save_fruit=True,
    compress=False,
    create_summary=True,
    split_hourly=False,
    gompertz_params=None,
    ci_mode="stomata",
    params=None,  # Assuming params dictionary is passed containing split_date, target_fruit_weight, etc.
    result_combined=None,  # Assuming this contains hourly/sub-daily data including Temperature and PAR
    leaves_info_df=None,  # Information about leaves that appeared
    removed_leaves_info_df=None,  # Information about removed leaves
    final_leaves_info_df=None,  # Hourly info on remaining leaves and their area per rank
    rank_photosynthesis_rates_df=None,  # Hourly gross photo rate per rank
    total_photosynthesis_df=None,  # Hourly total gross photo rate
    daily_photosynthesis_summary=None,  # Daily summary table to be filled
):
    """
    Simulate photosynthesis for a cucumber canopy with the given parameters.

    Args:
        file_path (str): Path to climate data file
        sheet_name (str): Sheet name in climate data file
        start_date (str): Start date for simulation (YYYY-MM-DD)
        initial_nodes (int): Initial number of nodes
        threshold_before (float): Thermal time threshold before split date
        threshold_after (float): Thermal time threshold after split date
        leaf_removal_dates_file (str): Path to leaf removal dates file
        leaf_removal_sheet_name (str): Sheet name in leaf removal dates file
        SLA (float): Specific leaf area (m^2/g)
        fruit_dw_file_path (str): Path to fruit dry weight data file
        fruit_dw_sheet_name (str): Sheet name in fruit dry weight data file
        partitioning_vegetative_before (float): Vegetative partitioning before fruit appearance
        partitioning_fruit_before (float): Fruit partitioning before fruit appearance
        initial_remaining_vegetative_dw (float): Initial vegetative dry weight
        plant_density_per_m2 (float): Plant density per square meter
        time_unit (str): Time unit for calculation ('minute' or 'hour')
        leaf_removal_mode (str): Mode for leaf removal ('file', 'interval', or 'threshold')
        leaf_removal_interval_days (int): Days between leaf removals (for interval mode)
        leaf_removal_time (str): Time of day for leaf removal (HH:MM:SS)
        leaf_removal_first_date (str): First date for interval-based removal (YYYY-MM-DD)
        leaf_removal_first_leaf_count (int): First leaf count to trigger removal (interval mode)
        storage_format (str): Format for storing results ('excel', 'csv', 'hdf5')
        output_dir (str): Directory to save results
        save_hourly (bool): Whether to save hourly data
        save_daily (bool): Whether to save daily data
        save_leaf (bool): Whether to save leaf-related data
        save_fruit (bool): Whether to save detailed fruit data
        compress (bool): Whether to compress output files
        create_summary (bool): Whether to create a summary of results
        split_hourly (bool): Whether to split hourly data into separate files
        gompertz_params (dict): Gompertz parameters for leaf removal calculation
        ci_mode (str): Method for Ci calculation ('stomata' or 'transpiration')
        params (dict): Parameters for flowering and fruit set logic
        result_combined (pandas.DataFrame): Combined result data including Temperature and PAR
        leaves_info_df (pandas.DataFrame): Information about leaves that appeared
        removed_leaves_info_df (pandas.DataFrame): Information about removed leaves
        final_leaves_info_df (pandas.DataFrame): Hourly info on remaining leaves and their area per rank
        rank_photosynthesis_rates_df (pandas.DataFrame): Hourly gross photo rate per rank
        total_photosynthesis_df (pandas.DataFrame): Hourly total gross photo rate
        daily_photosynthesis_summary (pandas.DataFrame): Daily summary table to be filled

    Returns:
        pandas.DataFrame: Daily photosynthesis summary
    """
    # Calculate time label based on unit
    time_label = TIME_UNITS[time_unit]["label"]

    # Initialize simulation with climate data and leaf removal dates
    initializer = SimulationInitializer(
        file_path,
        sheet_name,
        start_date,
        initial_nodes,
        threshold_before,
        threshold_after,
        params.get("split_date") if params else None,
        leaf_removal_dates_file,
        leaf_removal_sheet_name,
        leaf_removal_mode=leaf_removal_mode,
        leaf_removal_interval_days=leaf_removal_interval_days,
        leaf_removal_time=leaf_removal_time,
        leaf_removal_first_date=leaf_removal_first_date,
        leaf_removal_first_leaf_count=leaf_removal_first_leaf_count,
    )

    # Load climate data
    climate_data_results = initializer.initialize()
    if isinstance(climate_data_results, tuple) and len(climate_data_results) >= 4:
        df, hourly_climate_data, daily_avg_temp, leaf_removal_dates = climate_data_results
    else:
        print("Warning: Climate data not returned in expected format.")
        hourly_climate_data = climate_data_results
        leaf_removal_dates = pd.DataFrame(columns=["Date", "Time"])  # Empty fallback

        # Create empty daily_avg_temp from hourly data if needed
        if isinstance(hourly_climate_data, pd.DataFrame) and not hourly_climate_data.empty:
            try:
                daily_avg_temp = (
                    hourly_climate_data["Temperature"]
                    .groupby(pd.Grouper(freq="D"))
                    .mean()
                    .reset_index()
                )
                daily_avg_temp.columns = ["Date", "temperature"]
                daily_avg_temp.set_index("Date", inplace=True)
            except Exception as e:
                print(f"Warning: Could not create daily temperature data: {e}")
                daily_avg_temp = pd.DataFrame(columns=["Date", "temperature"])
                daily_avg_temp.set_index("Date", inplace=True)
        else:
            daily_avg_temp = pd.DataFrame(columns=["Date", "temperature"])
            daily_avg_temp.set_index("Date", inplace=True)

    # Correct conversion factor assuming Gompertz 'a' is in cm^2
    # Convert cm^2 per leaf to m^2 per m^2 ground area
    conversion_factor = plant_density_per_m2 / 10000.0

    # Initialize LeafCalculator
    split_date_from_params = params.get("split_date") if params else None
    leaf_calculator = LeafCalculator(
        df_hourly=hourly_climate_data,
        daily_avg_temp=daily_avg_temp,
        initial_nodes=initial_nodes,
        threshold_before=threshold_before,
        threshold_after=threshold_after,
        split_date=split_date_from_params,
        leaf_removal_dates=leaf_removal_dates,
        conversion_factor=conversion_factor,
        leaf_removal_config=LEAF_REMOVAL,
        leaf_removal_mode=leaf_removal_mode,
        a=gompertz_params["a"],
        b=gompertz_params["b"],
        c=gompertz_params["c"],
    )

    # Calculate leaves, nodes, and leaf removal
    result, leaves_info_df, removed_leaves_info_df = leaf_calculator.calculate_leaf_area()

    # *** 추가: 전달받은 removed_leaves_info_df 상태 확인 로그 ***
    logger.info(
        "--- Checking removed_leaves_info_df immediately after receiving from LeafCalculator ---"
    )
    if removed_leaves_info_df is not None:
        logger.info(f"Type: {type(removed_leaves_info_df)}")
        logger.info(f"Is empty: {removed_leaves_info_df.empty}")
        if not removed_leaves_info_df.empty:
            logger.info(f"Columns: {removed_leaves_info_df.columns.tolist()}")
            logger.info(f"Head:\n{removed_leaves_info_df.head().to_string()}")
            # Check specific columns needed later
            logger.info(
                f"Has 'Removal Date' column: {'Removal Date' in removed_leaves_info_df.columns}"
            )
            logger.info(
                f"Has 'Removed Leaf DW (g)' column: {'Removed Leaf DW (g)' in removed_leaves_info_df.columns}"
            )
        else:
            logger.info("DataFrame is empty.")
    else:
        logger.info("received_leaves_info_df is None.")
    logger.info("--- End check ---")

    # Get datetime column from index
    result = result.reset_index().rename(columns={"index": "DateTime"})
    result = result.set_index("DateTime")

    # Create leaves info DataFrames (Original leaves_info variable was from tuple, now using leaves_info_df)
    # leaves_info_df = pd.DataFrame(leaves_info) # This line seems redundant now as leaves_info_df is directly returned

    # Merge hourly climate data with leaf development results
    # Ensure index types match before concatenating
    if not isinstance(hourly_climate_data.index, pd.DatetimeIndex):
        hourly_climate_data.index = pd.to_datetime(hourly_climate_data.index)
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)

    # Align indexes before concat to avoid duplicate columns if indexes overlap partially
    aligned_climate, aligned_result = hourly_climate_data.align(result, join="outer", axis=0)
    result_combined = pd.concat([aligned_climate, aligned_result], axis=1)

    # Calculate remaining leaves info
    # Check if leaves_info_df is empty before proceeding
    if not leaves_info_df.empty:
        remaining_leaves_info = leaf_calculator.create_remaining_leaves_info(
            result_combined, leaves_info_df
        )
    else:
        remaining_leaves_info = []  # Handle case with no leaves

    # Check if remaining_leaves_info is empty
    if not remaining_leaves_info:
        print(
            "Warning: No remaining leaves info generated. Photosynthesis calculation will be skipped."
        )
        # Create empty dataframes to avoid errors later
        final_leaves_info_df = pd.DataFrame(columns=["Timestamp"]).set_index("Timestamp")
        rank_photosynthesis_rates_df = pd.DataFrame()
        total_photosynthesis_df = pd.DataFrame()
    else:
        # Transform remaining leaves info
        transformed_data = leaf_calculator.transform_remaining_leaves_info(remaining_leaves_info)

        # Pad transformed data
        padded_data, max_len = leaf_calculator.pad_transformed_data(transformed_data)

        # Create columns for final leaves DataFrame
        columns = leaf_calculator.create_columns(max_len)

        # Create final leaves DataFrame
        final_leaves_info_df = pd.DataFrame(padded_data, columns=columns)
        if "Timestamp" in final_leaves_info_df.columns:
            final_leaves_info_df["Timestamp"] = pd.to_datetime(final_leaves_info_df["Timestamp"])
            final_leaves_info_df.set_index("Timestamp", inplace=True)
        else:
            print("Warning: 'Timestamp' column missing after padding. Index cannot be set.")
            # Handle this case, maybe create an empty DataFrame
            final_leaves_info_df = pd.DataFrame(columns=["Timestamp"]).set_index("Timestamp")

    # Add environment data to result_combined
    result_combined["hs"] = 0.7  # Placeholder for humidity
    result_combined["Ci"] = (
        result_combined.get("CO2", 400) * 0.7
    )  # Placeholder for intercellular CO2

    if "PARi" not in result_combined.columns and "PAR" in result_combined.columns:
        result_combined["PARi"] = result_combined["PAR"]

    # --- Calculate Hourly LAI ---
    # Ensure final_leaves_info_df is not empty and has leaf area columns
    if not final_leaves_info_df.empty:
        # Find all leaf area columns
        leaf_area_cols = [
            col for col in final_leaves_info_df.columns if col.startswith("Leaf_Area_per_m2_")
        ]
        if leaf_area_cols:
            # Sum leaf areas for each timestamp (row), fill NaN with 0 before summing
            hourly_lai = final_leaves_info_df[leaf_area_cols].fillna(0).sum(axis=1)
            hourly_lai.name = "LAI"  # Name the series

            # Merge LAI into result_combined based on the index
            # Ensure both indexes are DatetimeIndex
            if not isinstance(result_combined.index, pd.DatetimeIndex):
                result_combined.index = pd.to_datetime(result_combined.index)
            if not isinstance(hourly_lai.index, pd.DatetimeIndex):
                hourly_lai.index = pd.to_datetime(hourly_lai.index)

            result_combined = pd.merge(
                result_combined, hourly_lai, left_index=True, right_index=True, how="left"
            )
            # Fill potential NaNs in LAI if merge resulted in missing rows
            if "LAI" in result_combined.columns:
                result_combined["LAI"] = result_combined["LAI"].fillna(0)
            else:
                print("Warning: LAI column was not added after merge.")
        else:
            print(
                "Warning: No 'Leaf_Area_per_m2_' columns found in final_leaves_info_df. Setting LAI to 0."
            )
            result_combined["LAI"] = 0.0
    else:
        print("Warning: final_leaves_info_df is empty. Setting LAI to 0.")
        result_combined["LAI"] = 0.0
    # --- End Hourly LAI Calculation ---

    # Calculate photosynthesis rates using the FvCB model
    if not final_leaves_info_df.empty:
        rank_photosynthesis_rates_df, total_photosynthesis_df = calculate_photosynthesis_dataframe(
            result_combined,
            final_leaves_info_df,
            plant_density_per_m2,
            time_unit=time_unit,
            ci_calculation_mode=ci_mode,
        )
    else:
        rank_photosynthesis_rates_df = pd.DataFrame()
        total_photosynthesis_df = pd.DataFrame()

    # Merge photosynthesis rates with result_combined
    if not total_photosynthesis_df.empty:
        result_combined = pd.merge(
            result_combined,
            total_photosynthesis_df,
            left_index=True,
            right_on="Timestamp",
            how="left",
        )
        result_combined = result_combined.set_index(result_combined.index)

    # Calculate maintenance respiration
    rm_df = calculate_rm(result_combined)

    # --- START: Dry Weight Calculation Logic ---
    # Aggregate total photosynthesis amount to daily level
    if not total_photosynthesis_df.empty:
        # Access the index directly and get the date part
        total_photosynthesis_df["Date"] = total_photosynthesis_df.index.date

        # Use label from TIME_UNITS config for consistency
        time_label = TIME_UNITS[time_unit]["label"]
        daily_photosynthesis_summary = (
            total_photosynthesis_df.groupby("Date")[
                f"total_gross_photosynthesis_amount/{time_label}"
            ]
            .sum()
            .reset_index()
        )
        daily_photosynthesis_summary.rename(
            columns={
                f"total_gross_photosynthesis_amount/{time_label}": f"total_gross_photosynthesis_amount/day (umol/m^2/day)"
            },
            inplace=True,
        )
    else:
        # Create an empty dataframe with expected columns if no photosynthesis occurred
        daily_photosynthesis_summary = pd.DataFrame(
            columns=["Date", f"total_gross_photosynthesis_amount/day (umol/m^2/day)"]
        )
        # Populate Date column based on simulation period
        if not result_combined.empty:
            all_dates = pd.to_datetime(result_combined.index).date
            unique_dates = sorted(list(set(all_dates)))
            daily_photosynthesis_summary["Date"] = unique_dates
            daily_photosynthesis_summary[
                f"total_gross_photosynthesis_amount/day (umol/m^2/day)"
            ] = 0.0

    # CH2O production calculation
    # Ensure the column name exists before calculation
    photo_col_name = f"total_gross_photosynthesis_amount/day (umol/m^2/day)"
    if photo_col_name in daily_photosynthesis_summary.columns:
        daily_photosynthesis_summary["A_grossCH2O_PRODUCTION (g/m^2/day)"] = (
            daily_photosynthesis_summary[photo_col_name] / 1_000_000
        ) * MOL_MASS_CH2O  # Simplified from * 44.01 * 30 / 44
    else:
        print(
            f"Warning: Column '{photo_col_name}' not found for CH2O calculation. Setting production to 0."
        )
        daily_photosynthesis_summary["A_grossCH2O_PRODUCTION (g/m^2/day)"] = 0.0

    # Convert Date in rm_df to date objects if they are Timestamps
    rm_df["Date"] = pd.to_datetime(rm_df["Date"]).dt.date
    daily_photosynthesis_summary["Date"] = pd.to_datetime(
        daily_photosynthesis_summary["Date"]
    ).dt.date

    # Merge Rm data
    daily_photosynthesis_summary = pd.merge(
        daily_photosynthesis_summary, rm_df, on="Date", how="left"
    )

    # Calculate Removed_Leaf_Area and Removed Leaf DW
    daily_removed_leaf_summary = pd.DataFrame(
        columns=["Date", "Removed_Leaf_Area", "Removed_Leaf_DW_g_per_m2"]
    )
    if (
        not removed_leaves_info_df.empty
        and "Removal Date" in removed_leaves_info_df.columns
        and "Removed Leaf DW (g)" in removed_leaves_info_df.columns
    ):
        try:
            logger.info("Processing removed leaf info...")
            # Ensure 'Removal Date' is in datetime format and extract date part
            removed_leaves_info_df["Removal Date_dt"] = pd.to_datetime(
                removed_leaves_info_df["Removal Date"], errors="coerce"
            )
            removed_leaves_info_df.dropna(subset=["Removal Date_dt"], inplace=True)
            removed_leaves_info_df["Removal_Date_Only"] = removed_leaves_info_df[
                "Removal Date_dt"
            ].dt.date

            # Ensure DW and Area columns are numeric
            removed_leaves_info_df["Removed Leaf Area (m2)"] = pd.to_numeric(
                removed_leaves_info_df["Removed Leaf Area (m2)"], errors="coerce"
            ).fillna(0)
            removed_leaves_info_df["Removed Leaf DW (g)"] = pd.to_numeric(
                removed_leaves_info_df["Removed Leaf DW (g)"], errors="coerce"
            ).fillna(0)

            logger.info(
                f"Removed leaves info (head) before grouping:\n{removed_leaves_info_df[['Removal_Date_Only', 'Removed Leaf Area (m2)', 'Removed Leaf DW (g)']].head().to_string()}"
            )
            logger.info(f"Data types of removed_leaves_info_df: \n{removed_leaves_info_df.dtypes}")

            # Group by date and sum Area and DW
            daily_removed_summary_grouped = (
                removed_leaves_info_df.groupby("Removal_Date_Only")[
                    [
                        "Removed Leaf Area (m2)",
                        "Removed Leaf DW (g)",
                    ]  # Use list for multiple columns
                ]
                .sum()
                .reset_index()
            )
            logger.info(
                f"Removed leaves info (head) after grouping:\n{daily_removed_summary_grouped.head().to_string()}"
            )

            # Calculate DW per m2 ground area
            logger.info(
                f"Using plant_density_per_m2 = {plant_density_per_m2} to calculate DW per m2."
            )
            if plant_density_per_m2 <= 0:
                logger.warning(
                    "plant_density_per_m2 is zero or negative. Removed DW per m2 will be zero."
                )
            daily_removed_summary_grouped["Removed_Leaf_DW_g_per_m2"] = (
                daily_removed_summary_grouped["Removed Leaf DW (g)"] * plant_density_per_m2
            )

            # Rename columns for merging
            daily_removed_summary_grouped.rename(
                columns={
                    "Removal_Date_Only": "Date",
                    "Removed Leaf Area (m2)": "Removed_Leaf_Area",
                },
                inplace=True,
            )

            # Select only needed columns and ensure 'Date' is the correct type (object, typically datetime.date)
            daily_removed_leaf_summary = daily_removed_summary_grouped[
                ["Date", "Removed_Leaf_Area", "Removed_Leaf_DW_g_per_m2"]
            ]
            # Explicitly check type just before merge if needed, but groupby usually preserves it
            logger.info(
                f"Final daily removed summary for merge (head):\n{daily_removed_leaf_summary.head().to_string()}"
            )
            logger.info(
                f"Data type of 'Date' column in daily_removed_leaf_summary: {daily_removed_leaf_summary['Date'].dtype}"
            )

        except Exception as e:
            logger.error(
                f"Error processing removed leaf info: {e}. Setting removed values to 0.",
                exc_info=True,
            )
            daily_removed_leaf_summary = pd.DataFrame(
                columns=["Date", "Removed_Leaf_Area", "Removed_Leaf_DW_g_per_m2"]
            )
    else:
        logger.info(
            "No removed leaves found or required columns missing. Setting removed values to 0."
        )
        daily_removed_leaf_summary = pd.DataFrame(
            columns=["Date", "Removed_Leaf_Area", "Removed_Leaf_DW_g_per_m2"]
        )

    # --- Prepare daily_photosynthesis_summary for Merge ---
    if "Date" not in daily_photosynthesis_summary.columns:
        logger.error("'Date' column missing from daily_photosynthesis_summary before merge.")
        if isinstance(daily_photosynthesis_summary.index, pd.DatetimeIndex):
            logger.warning("Creating 'Date' column from DatetimeIndex.")
            daily_photosynthesis_summary["Date"] = daily_photosynthesis_summary.index.date
        else:
            raise KeyError("Cannot merge: 'Date' column missing and index is not DatetimeIndex.")
    else:
        # Ensure 'Date' is datetime.date objects
        original_dtype = daily_photosynthesis_summary["Date"].dtype
        try:
            daily_photosynthesis_summary["Date"] = pd.to_datetime(
                daily_photosynthesis_summary["Date"], errors="coerce"
            ).dt.date
            daily_photosynthesis_summary.dropna(subset=["Date"], inplace=True)
            logger.info(
                f"Ensured 'Date' column in daily_photosynthesis_summary is type: {daily_photosynthesis_summary['Date'].dtype} (Original: {original_dtype})"
            )
        except Exception as e:
            logger.error(
                f"Failed to convert 'Date' column in daily_photosynthesis_summary to date objects: {e}",
                exc_info=True,
            )
            raise

    # --- Perform the Merge ---
    logger.info(
        f"Merging daily_photosynthesis_summary (shape {daily_photosynthesis_summary.shape}, Date type {daily_photosynthesis_summary['Date'].dtype}) with daily_removed_leaf_summary (shape {daily_removed_leaf_summary.shape}, Date type {daily_removed_leaf_summary['Date'].dtype})"
    )
    original_rows = len(daily_photosynthesis_summary)
    daily_photosynthesis_summary = pd.merge(
        daily_photosynthesis_summary, daily_removed_leaf_summary, on="Date", how="left"
    )
    if len(daily_photosynthesis_summary) != original_rows:
        logger.warning(
            f"Number of rows changed after merging removed leaf info ({original_rows} -> {len(daily_photosynthesis_summary)}). Check merge logic."
        )

    # Fill NaN values with 0 after merge
    daily_photosynthesis_summary["Removed_Leaf_Area"] = daily_photosynthesis_summary[
        "Removed_Leaf_Area"
    ].fillna(0)
    daily_photosynthesis_summary["Removed_Leaf_DW_g_per_m2"] = daily_photosynthesis_summary[
        "Removed_Leaf_DW_g_per_m2"
    ].fillna(0)

    # *** Log merged results ***
    logger.info("Checking values AFTER merge and fillna:")
    if (daily_photosynthesis_summary["Removed_Leaf_DW_g_per_m2"] > 1e-9).any():
        logger.info(
            daily_photosynthesis_summary[
                daily_photosynthesis_summary["Removed_Leaf_DW_g_per_m2"] > 1e-9
            ][["Date", "Removed_Leaf_Area", "Removed_Leaf_DW_g_per_m2"]]
            .head()
            .to_string()  # Show head instead of all
        )
    else:
        logger.info(
            "All Removed_Leaf_DW_g_per_m2 values are zero or very close to zero after merge and fillna."
        )

    # Add Harvested Fruit DW data
    if fruit_dw_file_path:
        try:  # Try to process harvest data
            fruit_dw_data = process_harvest_data(fruit_dw_file_path, sheet_name=fruit_dw_sheet_name)
            # Check if fruit_dw_data is not empty and contains 'Date' before processing
            if not fruit_dw_data.empty and "Date" in fruit_dw_data.columns:
                # Ensure 'Date' column is of type date
                fruit_dw_data["Date"] = pd.to_datetime(fruit_dw_data["Date"]).dt.date
                daily_photosynthesis_summary = add_fruit_dw_column(
                    daily_photosynthesis_summary, fruit_dw_data
                )
            else:
                # Add column with zeros if data is empty or 'Date' column is missing
                daily_photosynthesis_summary["Harvested Fruit DW (g/m^2)"] = 0.0
        except FileNotFoundError:
            # If file not found, silently treat as if no path was given
            daily_photosynthesis_summary["Harvested Fruit DW (g/m^2)"] = 0.0
        except Exception as e:
            # Still warn about other potential processing errors
            print(
                f"Warning: Error processing harvest data file {fruit_dw_file_path}: {e}. Setting harvested fruit DW to 0."
            )
            daily_photosynthesis_summary["Harvested Fruit DW (g/m^2)"] = 0.0
    else:
        # Add column with zeros if no harvest data file path provided
        daily_photosynthesis_summary["Harvested Fruit DW (g/m^2)"] = 0.0

    # Partitioning ratio calculation
    # Make sure daily_avg_temp index is date objects
    daily_avg_temp_copy = daily_avg_temp.copy()
    daily_avg_temp_copy.index = pd.to_datetime(daily_avg_temp_copy.index).date

    # Check if split_date is None, if so, set it far in the future to use 'before' logic
    safe_split_date = (
        pd.to_datetime(params.get("split_date")).date()
        if params.get("split_date")
        else pd.Timestamp.max.date()
    )

    daily_photosynthesis_summary["Partitioning to vegetative ratio"] = daily_photosynthesis_summary[
        "Date"
    ].apply(
        lambda date: (
            partitioning_vegetative_before
            if date < safe_split_date
            else 1 - (0.00786 * daily_avg_temp_copy.loc[date, "temperature"] + 0.2886)
        )
    )
    daily_photosynthesis_summary["Partitioning to fruit ratio"] = daily_photosynthesis_summary[
        "Date"
    ].apply(
        lambda date: (
            partitioning_fruit_before
            if date < safe_split_date
            else (0.00786 * daily_avg_temp_copy.loc[date, "temperature"] + 0.2886)
        )
    )

    # Initialize Dry Weight columns
    daily_photosynthesis_summary["Estimated Remaining Vegetative DW (g/m^2)"] = np.nan
    daily_photosynthesis_summary["Estimated Remaining fruit DW (g/m^2)"] = np.nan
    daily_photosynthesis_summary["CH2O for Maintenance Respiration (g/m^2/d)"] = np.nan
    daily_photosynthesis_summary["CH2O Available for Growth (Total-Rm) (g/m^2/d)"] = np.nan
    daily_photosynthesis_summary["Dry Matter Production (g/m^2/d)"] = np.nan
    daily_photosynthesis_summary["Estimated Vegetative DW Production (g/m^2/d)"] = np.nan
    daily_photosynthesis_summary["Estimated Fruit DW Production (g/m^2/d)"] = np.nan

    # --- Pre-calculate necessary daily data ---
    if not isinstance(result_combined.index, pd.DatetimeIndex):
        try:
            result_combined.index = pd.to_datetime(result_combined.index)
        except Exception as e:
            raise ValueError(f"Cannot convert result_combined index to DatetimeIndex: {e}")

    # === ADDED: Calculate Daily GDD and Cumulative GDD ===
    if "GDD" in result_combined.columns:
        daily_gdd = result_combined["GDD"].resample("D").sum()
        cumulative_daily_gdd = daily_gdd.cumsum()
        cumulative_daily_gdd.name = "Cumulative_GDD"
        # Ensure the index is date objects for merging
        cumulative_daily_gdd.index = cumulative_daily_gdd.index.date
        # Merge into daily_summary
        daily_photosynthesis_summary = pd.merge(
            daily_photosynthesis_summary,
            cumulative_daily_gdd,
            left_on="Date",
            right_index=True,
            how="left",
        )
        # Fill potential NaNs, although unlikely if dates align
        daily_photosynthesis_summary["Cumulative_GDD"] = (
            daily_photosynthesis_summary["Cumulative_GDD"].fillna(method="ffill").fillna(0)
        )
    else:
        logger.warning(
            "'GDD' column not found in result_combined. Cannot calculate Cumulative_GDD. Setting to 0."
        )
        daily_photosynthesis_summary["Cumulative_GDD"] = 0.0
    # ======================================================

    daily_avg_temp = result_combined["Temperature"].resample("D").mean()
    daily_photosynthesis_summary["Daily_Avg_Temp"] = daily_photosynthesis_summary["Date"].map(
        daily_avg_temp
    )

    if "PARi" in result_combined.columns:
        daily_par_wh = result_combined["PARi"].resample("D").sum()
        daily_par_mj = daily_par_wh * 0.0036
        daily_photosynthesis_summary["Daily_PAR_MJ"] = daily_photosynthesis_summary["Date"].map(
            daily_par_mj
        )
    else:
        # If PARi is not found, try PAR as a fallback, or raise error if neither found
        if "PAR" in result_combined.columns:
            print(
                "Warning: 'PARi' not found, using 'PAR' column instead for flowering calculation."
            )
            daily_par_wh = result_combined["PAR"].resample("D").sum()
            daily_par_mj = daily_par_wh * 0.0036
            daily_photosynthesis_summary["Daily_PAR_MJ"] = daily_photosynthesis_summary["Date"].map(
                daily_par_mj
            )
        else:
            raise KeyError(
                "Columns 'PARi' or 'PAR' not found in result_combined. Needed for flowering calculation."
            )

    if "total_nodes" not in result_combined.columns:
        raise KeyError("Column 'total_nodes' not found or calculation needed.")
    daily_total_nodes = result_combined["total_nodes"].resample("D").max().ffill()
    daily_photosynthesis_summary["total_nodes"] = (
        daily_photosynthesis_summary["Date"].map(daily_total_nodes).ffill().astype(int)
    )

    # --- Initialize state variables ---
    # flower_emergence_remainder = 0.0 # Replaced by flower_bud_potential_counter
    # flowering_nodes_waiting = [] # Replaced by node_states
    # fruit_set_info = {} # Replaced by node_states
    # node_waiting_info = {} # Replaced by node_states

    # New state management dictionary
    node_states = {}
    # Initialize states for initial nodes
    for i in range(1, initial_nodes + 1):
        node_states[i] = {
            "state": "appeared",
            "bud_init_date": None,
            "bud_init_node_count": None,
            "flowering_date": None,
            "consecutive_days_nr_below_one": 0,
            "set_date": None,
            "accum_dd": 0.0,
            "weight": 0.0,
            "appearance_date": (
                pd.to_datetime(start_date).date() if start_date else None
            ),  # Track appearance date
            "appearance_doy": (pd.to_datetime(start_date).dayofyear if start_date else None),
            "initial_cumulative_gdd": 0.0,
            "fruit_count": 0,
            "days_since_flowering": 0,
            "days_since_set": 0,
            "failure_reason": None,
        }
        if 1 <= i <= 9:
            node_states[i]["state"] = "thinned"
            logger.debug(f"Initial Node {i} marked as 'thinned'.")

    # Counter for potential flower buds based on N rate
    flower_bud_potential_counter = 0.0
    # Track nodes that appeared after split date for bud allocation
    nodes_appeared_after_split_date = []

    daily_fruit_details_log = []  # Keep for detailed logging if needed

    # Parse split_date
    split_date_str = params.get("split_date")
    if not split_date_str:
        raise ValueError("split_date is missing from the parameters.")
    try:
        split_date_obj = datetime.strptime(split_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid format/type for split_date: {split_date_str}. Expected YYYY-MM-DD string."
        )

    # Get target fruit weight
    default_target_fruit_weight = 150.0
    if "target_fruit_weight_per_node" not in params:
        logger.warning(
            f"'target_fruit_weight_per_node' not found. Using default: {default_target_fruit_weight}g"
        )
    TARGET_FRUIT_WEIGHT_PER_NODE = params.get(
        "target_fruit_weight_per_node", default_target_fruit_weight
    )

    # --- Add/Clean Columns in Daily Summary ---
    daily_photosynthesis_summary["Actual_Fruits_Set_Today"] = 0
    daily_photosynthesis_summary["Harvested_Fruit_DW_Today (g/m^2)"] = (
        0.0  # Internally calculated harvest
    )
    daily_photosynthesis_summary["Active_Fruit_Count"] = 0
    daily_photosynthesis_summary["Total_Fruit_DW (g/m^2)"] = 0.0
    daily_photosynthesis_summary["Permanently_Failed_Nodes_Today"] = 0  # Track mira-fruits
    daily_photosynthesis_summary["Cumulative_Failed_Nodes"] = 0  # Cumulative count of mira-fruits
    # Add daily node state counts
    daily_photosynthesis_summary["Nodes_State_Appeared"] = 0
    daily_photosynthesis_summary["Nodes_State_Bud"] = 0
    daily_photosynthesis_summary["Nodes_State_Flowering"] = 0
    daily_photosynthesis_summary["Nodes_State_Set"] = 0
    daily_photosynthesis_summary["Nodes_State_Harvested"] = 0
    daily_photosynthesis_summary["Nodes_State_Failed"] = 0
    daily_photosynthesis_summary["Nodes_State_Thinned"] = 0
    # Add columns for debugging/analysis
    daily_photosynthesis_summary["Source_SO (g/m^2/d)"] = 0.0
    daily_photosynthesis_summary["Sink_SI (g/m^2/d)"] = 0.0
    daily_photosynthesis_summary["SO_SI_Ratio"] = 0.0
    if "Estimated Remaining fruit DW (g/m^2)" in daily_photosynthesis_summary.columns:
        del daily_photosynthesis_summary["Estimated Remaining fruit DW (g/m^2)"]
    # ... (delete other old fruit columns) ...

    # --- Daily Calculation Loop ---
    if not isinstance(daily_photosynthesis_summary.index, pd.DatetimeIndex):
        daily_photosynthesis_summary.index = pd.to_datetime(daily_photosynthesis_summary["Date"])
    previous_total_nodes = initial_nodes
    nodes_eligible_for_bud = []  # Track nodes that appeared after split date and are ready for buds

    for date_index, daily_row in daily_photosynthesis_summary.iterrows():
        i = daily_photosynthesis_summary.index.get_loc(date_index)
        current_date = date_index.date()
        total_harvested_dw_today = 0.0

        # Get daily average temperature (used multiple times)
        daily_avg_temp = daily_row["Daily_Avg_Temp"]
        # === ADDED: Get current cumulative GDD ===
        current_cumulative_gdd = daily_row["Cumulative_GDD"]
        # =========================================
        if pd.isna(daily_avg_temp):
            logger.error(
                f"Date {current_date}: Cannot proceed with daily loop due to missing average temperature."
            )
            continue

        # --- 0. Update Fruit Age (Accumulated Degree Days) & Check Harvest BEFORE Sink Calc ---
        degree_days_today = max(0.0, daily_avg_temp - BASE_TEMP_FRUIT)
        nodes_to_harvest_today = []
        for node_rank, node_info in list(node_states.items()):  # Use list() for safe iteration
            if node_info["state"] == "set":
                node_info["accum_dd"] += degree_days_today
                logger.debug(
                    f"Date {current_date}: Updated accum_dd for node {node_rank} to {node_info['accum_dd']:.1f}"
                )
                # Check for harvest
                logger.debug(
                    f"Date {current_date}: Checking harvest for node {node_rank}. State: {node_info['state']}, Weight: {node_info['weight']:.4f}g, Target: {TARGET_FRUIT_WEIGHT_PER_NODE:.4f}g"
                )
                if node_info["weight"] >= TARGET_FRUIT_WEIGHT_PER_NODE:
                    nodes_to_harvest_today.append(node_rank)
                    total_harvested_dw_today += node_info["weight"]
                    logger.info(
                        f"Date {current_date}: Node {node_rank} harvest ({node_info['weight']:.2f}g). Fruit age: {node_info['accum_dd']:.1f} DD"
                    )
        # Update state for harvested nodes
        for node_rank in nodes_to_harvest_today:
            if node_rank in node_states:
                node_states[node_rank]["state"] = "harvested"
        daily_photosynthesis_summary.loc[date_index, "Harvested_Fruit_DW_Today (g/m^2)"] = (
            total_harvested_dw_today
        )

        # --- 1. Node Appearance Check ---
        current_total_nodes = int(daily_row["total_nodes"])
        newly_appeared_nodes = []  # List to store ranks of nodes that appeared today
        if current_total_nodes > previous_total_nodes:
            for node_rank in range(previous_total_nodes + 1, current_total_nodes + 1):
                if node_rank not in node_states:
                    node_states[node_rank] = {
                        "state": "appeared",
                        "bud_init_date": None,
                        "bud_init_node_count": None,
                        "flowering_date": None,
                        "consecutive_days_nr_below_one": 0,
                        "set_date": None,
                        "accum_dd": 0.0,
                        "weight": 0.0,
                        "appearance_date": current_date,
                        "appearance_doy": current_date.timetuple().tm_yday,
                        "initial_cumulative_gdd": current_cumulative_gdd,
                        "fruit_count": 0,
                        "days_since_flowering": 0,
                        "days_since_set": 0,
                        "failure_reason": None,
                    }
                    newly_appeared_nodes.append(node_rank)  # Add rank to the list

                    if 1 <= node_rank <= 9:
                        node_states[node_rank]["state"] = "thinned"
                        logger.debug(
                            f"Date {current_date}: Node {node_rank} appeared and marked as 'thinned'."
                        )
                    else:
                        logger.debug(f"Date {current_date}: Node {node_rank} appeared.")
                        if current_date >= split_date_obj:
                            nodes_eligible_for_bud.append(node_rank)
                            logger.debug(
                                f"Date {current_date}: Node {node_rank} added to bud eligibility list."
                            )
        previous_total_nodes = current_total_nodes
        # Store the count of newly appeared nodes for the daily summary
        newly_appeared_count_today = len(newly_appeared_nodes)
        daily_photosynthesis_summary.loc[date_index, "Newly_Appeared_Nodes_Today"] = (
            newly_appeared_count_today
        )

        # Sort eligibility list by node rank (appearance order)
        nodes_eligible_for_bud.sort()

        # --- 1.5 Flower Bud Formation (Starts from day 1, affects eligible nodes) ---
        # This section now runs every day if conditions are met
        daily_par_mj = daily_row["Daily_PAR_MJ"]
        if pd.isna(daily_par_mj):
            logger.warning(f"Date {current_date}: Skipping flower bud calc due to missing PAR.")
        else:
            # Calculate potential flower buds (N)
            rate_n = calculate_flower_appearance_rate(daily_avg_temp, daily_par_mj)
            flower_bud_potential_counter += rate_n
            logger.debug(
                f"Date {current_date}: N={rate_n:.3f}, Bud Potential Counter = {flower_bud_potential_counter:.3f}"
            )

            buds_formed_today = 0
            # Get all nodes currently in 'appeared' state, sorted by rank (appearance order)
            appeared_nodes_eligible = sorted(
                [rank for rank, info in node_states.items() if info["state"] == "appeared"]
            )

            # Allocate buds based on accumulated potential to eligible 'appeared' nodes
            for node_rank in appeared_nodes_eligible:
                if flower_bud_potential_counter >= 1.0:
                    # Check again in case state changed within the loop (shouldn't happen here)
                    if node_rank in node_states and node_states[node_rank]["state"] == "appeared":
                        node_states[node_rank]["state"] = "bud"
                        node_states[node_rank]["bud_init_date"] = current_date
                        node_states[node_rank][
                            "bud_init_node_count"
                        ] = current_total_nodes  # Node count when bud formed
                        flower_bud_potential_counter -= 1.0
                        # We don't need nodes_eligible_for_bud list anymore for this logic
                        buds_formed_today += 1
                        logger.info(
                            f"Date {current_date}: Flower bud formed on node {node_rank}. Counter = {flower_bud_potential_counter:.3f}"
                        )
                else:
                    break  # Stop allocating if potential drops below 1
            # Log if potential remains but no eligible nodes
            if (
                flower_bud_potential_counter >= 1.0
                and buds_formed_today == 0
                and not appeared_nodes_eligible  # Check the correct list
            ):
                logger.warning(
                    f"Date {current_date}: Bud potential {flower_bud_potential_counter:.3f} remains, but no nodes in 'appeared' state."
                )

        # --- 1.7 Flowering Check (Still gated by total_nodes >= 10) ---
        nodes_flowering_today = []
        # Only check for flowering if the plant has at least 10 nodes
        if current_total_nodes >= 10:
            for node_rank, node_info in node_states.items():
                if node_info["state"] == "bud":
                    # REVERTED: Check if 5 subsequent nodes have appeared (was incorrectly changed to 3)
                    if current_total_nodes >= node_info["bud_init_node_count"] + 5:
                        node_info["state"] = "flowering"
                        node_info["flowering_date"] = current_date
                        node_info["consecutive_days_nr_below_one"] = 0  # Reset counter on flowering
                        nodes_flowering_today.append(node_rank)
                        logger.info(f"Date {current_date}: Node {node_rank} is now flowering.")
        else:
            # Log if flowering check is skipped due to node count
            logger.debug(
                f"Date {current_date}: Skipping flowering check as total nodes ({current_total_nodes}) is less than 10."
            )

        # --- 2. Calculate Sink Strength (SI) ---
        # Vegetative Sink Strength
        pot_veg_growth_per_plant = calculate_potential_veg_growth(daily_avg_temp)
        veg_sink_strength_total = pot_veg_growth_per_plant * plant_density_per_m2

        # Fruit Sink Strength (Based on potential growth of 'set' fruits)
        total_fruit_sink_strength_per_plant = 0.0
        potential_fruit_growths = {}  # Store Y_pot_i for partitioning later
        for node_rank, node_info in node_states.items():
            if node_info["state"] == "set":
                accum_dd = node_info["accum_dd"]
                # Use the existing function for potential fruit growth rate
                pot_fruit_growth = calculate_potential_fruit_growth(accum_dd, daily_avg_temp)
                potential_fruit_growths[node_rank] = pot_fruit_growth
                total_fruit_sink_strength_per_plant += pot_fruit_growth
                logger.debug(
                    f"Date {current_date}: Node {node_rank} (State: Set, X_i={accum_dd:.1f}): Potential growth Y_pot={pot_fruit_growth:.3f} g/fruit/day"
                )

        fruit_sink_strength_total = total_fruit_sink_strength_per_plant * plant_density_per_m2

        # Total Sink Strength (SI)
        total_sink_si = veg_sink_strength_total + fruit_sink_strength_total
        daily_photosynthesis_summary.loc[date_index, "Sink_SI (g/m^2/d)"] = total_sink_si
        logger.debug(
            f"Date {current_date}: Total Sink SI = {veg_sink_strength_total:.3f} (veg) + {fruit_sink_strength_total:.3f} (fruit) = {total_sink_si:.3f} g/m2/d"
        )

        # --- 3. Calculate Source (SO) and SO/SI Ratio ---
        # Maintenance respiration calculation depends on previous day's biomass
        if i == 0:
            current_veg_dw = initial_remaining_vegetative_dw
            current_total_fruit_dw = sum(
                info["weight"] for info in node_states.values()
            )  # Should be 0 initially
        else:
            prev_row_index = daily_photosynthesis_summary.index[i - 1]
            prev_veg_dw = daily_photosynthesis_summary.loc[
                prev_row_index, "Estimated Remaining Vegetative DW (g/m^2)"
            ]
            prev_total_fruit_dw = daily_photosynthesis_summary.loc[
                prev_row_index, "Total_Fruit_DW (g/m^2)"
            ]

        rm_veg = daily_row["Rm_Vegetative (CH2Og/g DM)"]
        rm_fruit = daily_row["Rm_Fruit (CH2Og/g DM)"]
        if i == 0:
            maintenance_resp = (current_veg_dw * rm_veg) + (current_total_fruit_dw * rm_fruit)
        else:
            maintenance_resp = (prev_veg_dw * rm_veg) + (prev_total_fruit_dw * rm_fruit)

        gross_ch2o_prod = daily_row["A_grossCH2O_PRODUCTION (g/m^2/day)"]
        total_rm = gross_ch2o_prod - maintenance_resp  # Net CH2O available
        dm_prod = total_rm / GROWTH_EFFICIENCY  # Actual DM production today (Source, SO)
        dm_prod = max(0.0, dm_prod)  # Ensure non-negative source

        daily_photosynthesis_summary.loc[date_index, "Source_SO (g/m^2/d)"] = dm_prod

        # Calculate SO/SI Ratio
        if total_sink_si > 1e-9:  # Avoid division by zero
            so_si_ratio = dm_prod / total_sink_si
        else:
            # If SI is zero (e.g., very cold, no fruits), how to define ratio?
            # If SO is also zero, ratio is undefined (maybe 1?). If SO > 0, ratio is infinite (maybe clamp?).
            so_si_ratio = (
                1.0 if dm_prod < 1e-9 else 10.0
            )  # Example: Clamp high if SO>0, SI=0; set to 1 if both=0
            logger.warning(
                f"Date {current_date}: Total Sink SI is near zero ({total_sink_si:.2e}). SO/SI set to {so_si_ratio}"
            )

        # TODO: Implement 5-day average logic for SO/SI if required by paper.
        # For now, use the calculated daily ratio.
        final_so_si_ratio_for_nr = so_si_ratio
        daily_photosynthesis_summary.loc[date_index, "SO_SI_Ratio"] = final_so_si_ratio_for_nr
        logger.info(
            f"Date {current_date}: SO={dm_prod:.3f}, SI={total_sink_si:.3f}, SO/SI Ratio={final_so_si_ratio_for_nr:.3f}"
        )

        # --- 4. Flowering and Fruit Set Calculation -> Renamed to NR Calc & Mira-Fruit Check ---
        daily_par_mj = daily_row["Daily_PAR_MJ"]  # Already retrieved earlier if needed
        # actual_fruits_set_today = 0 # This is determined later based on NR and flowering nodes
        current_nr = 0.0  # Track current NR value
        if pd.isna(daily_par_mj):
            # NR calculation might still proceed if only PAR is missing? Check dependencies.
            # Assuming NR depends only on SO/SI and Temp for now.
            potential_nr = calculate_potential_fruit_set(final_so_si_ratio_for_nr, daily_avg_temp)
            current_nr = potential_nr
            logger.warning(f"Date {current_date}: Calculating NR without PAR data.")
        else:
            # rate_n = calculate_flower_appearance_rate(daily_avg_temp, daily_par_mj) # N is calculated earlier for buds
            potential_nr = calculate_potential_fruit_set(final_so_si_ratio_for_nr, daily_avg_temp)
            current_nr = potential_nr
            logger.info(
                f"Date {current_date}: Potential NR calculated: {current_nr:.3f}"
            )  # Log NR value

        # --- Check for Failure based on Duration in 'flowering' state --- MODIFIED LOGIC
        nodes_failed_today = []
        permanently_failed_nodes_details = []

        for node_rank, node_info in node_states.items():
            if node_info["state"] == "flowering":
                # Calculate days elapsed since flowering started
                if node_info["flowering_date"]:
                    try:
                        days_in_flowering = (current_date - node_info["flowering_date"]).days
                    except TypeError as e:
                        logger.error(
                            f"Date {current_date}: Error calculating days in flowering for node {node_rank}. Flowering date: {node_info['flowering_date']}. Error: {e}"
                        )
                        days_in_flowering = -1  # Indicate error
                else:
                    logger.warning(
                        f"Date {current_date}: Node {node_rank} is 'flowering' but has no flowering_date. Cannot check failure duration."
                    )
                    days_in_flowering = -1

                # Check if node has been flowering for 3+ days (Changed from 5)
                if days_in_flowering >= 2:
                    logger.info(
                        f"Date {current_date}: Node {node_rank} marked as failed. "
                        # Updated log message to reflect 3 days
                        f"Remained in 'flowering' state for {days_in_flowering} days (>= 2). "
                        f"Flowered on: {node_info['flowering_date']}."
                    )
                    node_info["state"] = "failed"  # Mark as permanently failed
                    nodes_failed_today.append(node_rank)
                    permanently_failed_nodes_details.append(
                        {
                            "Date": current_date,
                            "NodeRank": node_rank,
                            "Status": "PermanentlyFailedSet_Duration",  # Indicate new reason
                            "FloweringDate": node_info.get("flowering_date"),
                            "DaysInFlowering": days_in_flowering,  # Store duration instead of NR counter
                            # "ConsecutiveDaysNRBelowOne": node_info["consecutive_days_nr_below_one"], # Removed old counter
                        }
                    )

        # Update the daily count of permanently failed nodes
        daily_photosynthesis_summary.loc[date_index, "Permanently_Failed_Nodes_Today"] = len(
            nodes_failed_today
        )
        # Update cumulative count (keep existing logic, but use nodes_failed_today)
        if i == 0:
            daily_photosynthesis_summary.loc[date_index, "Cumulative_Failed_Nodes"] = len(
                nodes_failed_today
            )
        else:
            prev_row_index = daily_photosynthesis_summary.index[i - 1]
            prev_cumulative = daily_photosynthesis_summary.loc[
                prev_row_index, "Cumulative_Failed_Nodes"
            ]
            daily_photosynthesis_summary.loc[date_index, "Cumulative_Failed_Nodes"] = (
                prev_cumulative + len(nodes_failed_today)
            )

        # --- 5. Assign Fruit Set (Based on NR and 'flowering' nodes with rank >= 10) ---
        nodes_set_this_day = []
        assigned_count = 0
        max_settable_fruits = math.floor(max(0.0, current_nr))  # Max fruits to set today

        # Get nodes eligible for setting fruit ('flowering' state AND node rank >= 10, sorted)
        eligible_setting_nodes = sorted(
            [
                rank
                for rank, info in node_states.items()
                if info["state"] == "flowering"
                and rank >= 10  # ADDED: Only consider nodes from rank 10 onwards
            ]
        )

        logger.debug(
            f"Date {current_date}: Max settable fruits (NR floor): {max_settable_fruits}. Eligible flowering nodes (Rank >= 10): {eligible_setting_nodes}"
        )

        for node_rank in eligible_setting_nodes:
            if assigned_count < max_settable_fruits:
                if node_rank in node_states:  # Check exists
                    # Check state again, might have changed to 'failed' just before this
                    if node_states[node_rank]["state"] == "flowering":
                        node_states[node_rank]["state"] = "set"
                        node_states[node_rank]["set_date"] = current_date
                        node_states[node_rank][
                            "accum_dd"
                        ] = degree_days_today  # Start accumulating DD from today
                        node_states[node_rank]["weight"] = 0.0  # Start weight at 0
                        node_states[node_rank]["consecutive_days_nr_below_one"] = 0  # Reset counter

                        nodes_set_this_day.append(node_rank)
                        assigned_count += 1
                        logger.info(
                            f"Date {current_date}: Fruit set assigned to node {node_rank}. Initial DD: {degree_days_today:.1f}"
                        )
                    else:
                        logger.warning(
                            f"Date {current_date}: Node {node_rank} was eligible for set but state was '{node_states[node_rank]['state']}' (likely failed just before). Skipping set."
                        )

            else:
                break  # Stop if max settable fruits reached

        daily_photosynthesis_summary.loc[date_index, "Actual_Fruits_Set_Today"] = assigned_count
        if assigned_count < max_settable_fruits and eligible_setting_nodes:
            logger.warning(
                f"Date {current_date}: Wanted to set {max_settable_fruits} fruits (NR), but only {len(eligible_setting_nodes)} nodes were in 'flowering' state. Set {assigned_count}."
            )

        # --- 6. DM Partitioning (Source term already calculated as dm_prod) ---
        part_veg_ratio = daily_row["Partitioning to vegetative ratio"]
        part_fruit_ratio = daily_row["Partitioning to fruit ratio"]
        veg_dw_prod = dm_prod * part_veg_ratio
        total_fruit_dw_prod = dm_prod * part_fruit_ratio

        # --- 7. Distribute Fruit DM Production (Based on relative potential sink strength) ---
        # potential_fruit_growths dictionary was calculated in Step 2 (SI calculation)
        total_potential_fruit_sink = sum(potential_fruit_growths.values())
        num_active_fruits = len(potential_fruit_growths)

        if num_active_fruits > 0 and total_fruit_dw_prod > 1e-9:
            if total_potential_fruit_sink > 1e-9:  # Check if total potential sink is positive
                # Distribute based on relative sink strength
                for node_rank, potential_growth in potential_fruit_growths.items():
                    if node_rank in node_states:  # Ensure node still exists
                        relative_sink = potential_growth / total_potential_fruit_sink
                        dw_allocated_today = total_fruit_dw_prod * relative_sink
                        node_states[node_rank]["weight"] += dw_allocated_today
                        logger.debug(
                            f"Date {current_date}: Allocated {dw_allocated_today:.3f}g DW (RelSink={relative_sink:.3f}) to node {node_rank}. New weight: {node_states[node_rank]['weight']:.3f}g"
                        )
            else:
                # If total potential sink is zero (e.g., very cold), distribute equally or log warning?
                dw_per_fruit_today = total_fruit_dw_prod / num_active_fruits
                logger.warning(
                    f"Date {current_date}: Total potential fruit sink is zero. Distributing {total_fruit_dw_prod:.3f}g equally among {num_active_fruits} fruits."
                )
                for node_rank in potential_fruit_growths.keys():
                    if node_rank in node_states:
                        node_states[node_rank]["weight"] += dw_per_fruit_today
                        logger.debug(
                            f"Date {current_date}: Allocated {dw_per_fruit_today:.3f}g DW (Equal) to node {node_rank}. New weight: {node_states[node_rank]['weight']:.3f}g"
                        )
        elif total_fruit_dw_prod > 1e-9:
            logger.warning(
                f"Date {current_date}: {total_fruit_dw_prod:.3f}g fruit DM available, but no active fruits in 'set' state."
            )

        # --- 8. Check for Harvest --- (Moved to beginning of loop - Step 0)
        # nodes_to_harvest = []
        # ... harvest logic removed from here ...

        # --- 9. Update Remaining Vegetative DW ---
        removed_leaf_dw = daily_row["Removed_Leaf_DW_g_per_m2"]
        if i == 0:  # First day
            remaining_veg_dw = initial_remaining_vegetative_dw + veg_dw_prod - removed_leaf_dw
        else:
            prev_row_index = daily_photosynthesis_summary.index[i - 1]
            prev_veg_dw = daily_photosynthesis_summary.loc[
                prev_row_index, "Estimated Remaining Vegetative DW (g/m^2)"
            ]
            remaining_veg_dw = prev_veg_dw + veg_dw_prod - removed_leaf_dw
        remaining_veg_dw = max(0, remaining_veg_dw)
        daily_photosynthesis_summary.loc[
            date_index, "Estimated Remaining Vegetative DW (g/m^2)"
        ] = remaining_veg_dw

        # --- 10. Update Total Remaining Fruit DW ---
        current_total_fruit_dw = sum(
            info["weight"] for info in node_states.values() if info["state"] == "set"
        )
        current_total_fruit_dw = max(0, current_total_fruit_dw)
        daily_photosynthesis_summary.loc[date_index, "Total_Fruit_DW (g/m^2)"] = (
            current_total_fruit_dw
        )

        # --- 10.5 Update Active Fruit Count ---
        active_fruit_count = sum(1 for info in node_states.values() if info["state"] == "set")
        daily_photosynthesis_summary.loc[date_index, "Active_Fruit_Count"] = active_fruit_count

        # --- 11. Store other daily results ---
        daily_photosynthesis_summary.loc[
            date_index, "CH2O for Maintenance Respiration (g/m^2/d)"
        ] = maintenance_resp
        daily_photosynthesis_summary.loc[
            date_index, "CH2O Available for Growth (Total-Rm) (g/m^2/d)"
        ] = total_rm
        daily_photosynthesis_summary.loc[date_index, "Dry Matter Production (g/m^2/d)"] = (
            dm_prod  # This is SO
        )
        daily_photosynthesis_summary.loc[
            date_index, "Estimated Vegetative DW Production (g/m^2/d)"
        ] = veg_dw_prod
        # Maybe log Estimated Fruit DW Production based on allocated amounts?
        daily_photosynthesis_summary.loc[date_index, "Estimated Fruit DW Production (g/m^2/d)"] = (
            total_fruit_dw_prod  # Total allocated
        )

        # --- 12. Log Daily Fruit Details --- # Using node_states
        current_day_details = []

        # Add permanently failed node records saved earlier
        current_day_details.extend(permanently_failed_nodes_details)

        # Log nodes in various states
        for node_rank, info in node_states.items():
            detail = {"Date": current_date, "NodeRank": node_rank, "Status": info["state"]}
            if info["state"] == "set":
                detail["FruitDW_g"] = info["weight"]
                detail["AccumDD_Cday"] = info["accum_dd"]
            elif info["state"] == "bud":
                detail["BudInitDate"] = info["bud_init_date"]
                detail["BudInitNodeCount"] = info["bud_init_node_count"]
            elif info["state"] == "flowering":
                detail["FloweringDate"] = info["flowering_date"]
                detail["ConsecutiveDaysNRBelowOne"] = info["consecutive_days_nr_below_one"]
            # Add other states if needed
            current_day_details.append(detail)

        # # Log nodes that successfully set fruit TODAY (already covered by state logging)
        # for node_rank in nodes_set_this_day:
        #     ...

        # # Log nodes that failed to set fruit today (now covered by 'flowering' state log)
        # failed_to_set_today = ...
        # for node_rank in failed_to_set_today:
        #     ...

        daily_fruit_details_log.extend(current_day_details)

        # --- 13. Update Daily Node State Counts in Summary ---
        state_counts = {
            state: 0 for state in ["bud", "flowering", "set", "harvested", "failed", "thinned"]
        }
        nodes_in_set_state_debug = []  # Debug list for 'set' nodes

        # Count nodes in each relevant state
        for node_rank, node_info in node_states.items():
            node_state = node_info["state"]
            if node_state != "appeared":  # Exclude nodes that just appeared
                if node_state in state_counts:
                    state_counts[node_state] += 1
                    # Debug: Track nodes counted as 'set'
                    if node_state == "set":
                        nodes_in_set_state_debug.append(node_rank)

        # Log the debug information for 'set' state count
        logger.debug(
            f"Date {current_date}: Nodes counted in 'set' state: {nodes_in_set_state_debug}. Final count: {state_counts['set']}"
        )

        # Store the counts in the daily summary DataFrame
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Bud"] = state_counts["bud"]
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Flowering"] = state_counts[
            "flowering"
        ]
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Set"] = state_counts["set"]
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Harvested"] = state_counts[
            "harvested"
        ]
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Failed"] = state_counts["failed"]
        daily_photosynthesis_summary.loc[date_index, "Nodes_State_Thinned"] = state_counts[
            "thinned"
        ]

        # Store cumulative failed nodes count
        daily_photosynthesis_summary.loc[date_index, "Permanently_Failed_Nodes_Today"] = len(
            nodes_failed_today
        )
        # Ensure the column exists before trying to get the previous value
        if "Cumulative_Failed_Nodes" not in daily_photosynthesis_summary.columns:
            daily_photosynthesis_summary["Cumulative_Failed_Nodes"] = 0

        previous_cumulative_failed = (
            daily_photosynthesis_summary["Cumulative_Failed_Nodes"].iloc[-2]
            if len(daily_photosynthesis_summary) > 1
            else 0
        )
        daily_photosynthesis_summary.loc[date_index, "Cumulative_Failed_Nodes"] = (
            previous_cumulative_failed + len(nodes_failed_today)
        )

    # Convert the log into a DataFrame # *** NEW STEP ***
    daily_fruit_details_df = pd.DataFrame(daily_fruit_details_log)

    # Convert final node states to DataFrame
    node_final_states_df = pd.DataFrame.from_dict(node_states, orient="index")
    node_final_states_df.index.name = "NodeRank"  # Set index name for clarity

    # --- Final Result Preparation ---
    results = {
        "hourly_data": result_combined,
        "daily_summary": daily_photosynthesis_summary,
        "leaves_info": leaves_info_df,
        "removed_leaves_info": removed_leaves_info_df,
        "final_leaves_hourly": final_leaves_info_df,
        "rank_photosynthesis_hourly": rank_photosynthesis_rates_df,
        "total_photosynthesis_hourly": total_photosynthesis_df,
        "fruit_details_daily": daily_fruit_details_df,
        "Node_Final_States": node_final_states_df,  # Use the created DataFrame with a descriptive key
    }

    # --- Save Results ---
    if output_dir:
        # Ensure base_filename is defined if needed, or handled by save_simulation_results
        # base_filename = "simulation" # Example base filename, adjust as necessary - Removed as function doesn't expect it
        saved_path = save_simulation_results(  # Capture the returned path
            results,
            output_dir=output_dir,
            # base_filename=base_filename, # Pass the base filename - REMOVED
            save_hourly=save_hourly,
            save_daily=save_daily,
            save_leaf=save_leaf,
            save_fruit=save_fruit,
            split_hourly_files=split_hourly,  # Changed back to original parameter name
            time_unit=time_unit,
            storage_format=storage_format,
            compress_output=compress,  # Changed back to original parameter name
            create_summary=create_summary,  # Changed back to original parameter name
        )
        if saved_path:  # Check if saving was successful
            logger.info(f"Simulation results saved to directory/file: {saved_path}")
        else:
            logger.error("Failed to save simulation results.")
    else:
        logger.warning("Output directory not specified. Results will not be saved.")

    return results

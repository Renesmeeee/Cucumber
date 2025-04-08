"""
Leaf Area Calculation Module
"""

import pandas as pd
import numpy as np
from src.utils.function import gompertz_growth
from src.configs.params import GOMPERTZ_PARAMETERS, LEAF_REMOVAL, DEFAULT_SETTINGS
from src.management.leaf_removal import process_leaf_removal
import logging

logger = logging.getLogger("leaf_area")


class LeafCalculator:
    """
    Calculator for leaf growth, development and removal processes
    """

    def __init__(
        self,
        df_hourly,
        daily_avg_temp,
        initial_nodes,
        threshold_before,
        threshold_after,
        split_date,
        leaf_removal_dates,
        conversion_factor,
        leaf_removal_config,
        leaf_removal_mode="file",
        a=GOMPERTZ_PARAMETERS["a"],
        b=GOMPERTZ_PARAMETERS["b"],
        c=GOMPERTZ_PARAMETERS["c"],
        SLA=DEFAULT_SETTINGS["SLA"],
    ):
        """
        Initialize the LeafCalculator

        Args:
            df_hourly (DataFrame): Hourly environmental data
            daily_avg_temp (DataFrame): Daily average temperature data
            initial_nodes (int): Initial number of nodes
            threshold_before (float): Thermal time threshold before split date
            threshold_after (float): Thermal time threshold after split date
            split_date (str): Date when threshold changes
            leaf_removal_dates (DataFrame): Dates when leaves are removed
            conversion_factor (float): Conversion factor for leaf area
            leaf_removal_config (dict): Configuration for leaf removal
            leaf_removal_mode (str): Mode for leaf removal: 'file', 'interval', or 'threshold'
            a (float): Gompertz parameter a
            b (float): Gompertz parameter b
            c (float): Gompertz parameter c
            SLA (float): SLA parameter
        """
        self.df_hourly = df_hourly
        self.daily_avg_temp = daily_avg_temp
        self.initial_nodes = initial_nodes
        self.threshold_before = threshold_before
        self.threshold_after = threshold_after
        self.split_date = pd.to_datetime(split_date).date()
        self.leaf_removal_dates = leaf_removal_dates
        self.conversion_factor = conversion_factor
        self.leaf_removal_mode = leaf_removal_mode
        self.leaf_removal_config = leaf_removal_config
        self.a = a
        self.b = b
        self.c = c
        self.first_removal_done = False
        self.end_date = df_hourly.index.max()
        self.removed_leaves = []  # To store info about removed leaves
        # self._load_data() # Delay loading until needed
        self.max_leaf_no = 0
        self.SLA = SLA

    def calculate_leaf_area(self):
        """
        Calculate leaf area based on thermal time and handle leaf removal.

        Returns:
            tuple: result (DataFrame), leaves_info (DataFrame), removed_leaves_info (DataFrame)
        """
        cumulative_sum = 0
        nodes = self.initial_nodes
        remaining_leaves = self.initial_nodes
        leaves_info = []  # List to store active leaf dictionaries
        removed_leaves_info = []  # List to store details of removed leaves (for final DataFrame)
        cumulative_thermal_time = 0
        all_leaves_history = []  # List to store the history of ALL leaves that appeared

        result = pd.DataFrame(
            index=pd.date_range(
                start=self.df_hourly.index.min(), end=self.df_hourly.index.max(), freq="H"
            )
        )
        result["remaining_leaves"] = np.nan
        result["total_nodes"] = np.nan
        result["cumulative_thermal_time_until_yesterday"] = np.nan
        result["cumulative_thermal_time"] = np.nan

        simulation_start_date = self.df_hourly.index[0].normalize()
        for leaf_number in range(1, self.initial_nodes + 1):
            leaves_info.append(
                {
                    "Leaf Number": leaf_number,
                    "Date": simulation_start_date,  # Use normalized start date
                    "Thermal Time": 0,
                }
            )
            # Also add initial leaves to the history
            all_leaves_history.append(
                {
                    "Leaf Number": leaf_number,
                    "Date": simulation_start_date,
                    "Thermal Time": 0,
                }
            )

        leaf_number_counter = self.initial_nodes  # Use a separate counter for new leaf numbers

        # Use a copy for modification if interval mode generates dates
        current_leaf_removal_dates = self.leaf_removal_dates.copy()

        for i in range(len(self.daily_avg_temp)):
            current_date = self.daily_avg_temp.index[i]
            daily_growing_temp = self.daily_avg_temp["daily_growing_temp"].iloc[i]

            threshold = (
                self.threshold_before if current_date < self.split_date else self.threshold_after
            )

            # Store cumulative thermal time from the beginning of the day
            day_start_cumulative_tt = cumulative_thermal_time

            cumulative_sum += daily_growing_temp
            while cumulative_sum >= threshold:  # Use while loop for multiple leaves per day
                nodes += 1
                # remaining_leaves += 1 # This is handled by the length of leaves_info after removal
                cumulative_sum -= threshold
                leaf_number_counter += 1
                leaves_info.append(
                    {
                        "Leaf Number": leaf_number_counter,
                        "Date": pd.Timestamp(current_date),  # Ensure Date is Timestamp
                        "Thermal Time": day_start_cumulative_tt,  # TT when the leaf appears
                    }
                )
                # Add to history as well
                all_leaves_history.append(
                    {
                        "Leaf Number": leaf_number_counter,
                        "Date": pd.Timestamp(current_date),
                        "Thermal Time": day_start_cumulative_tt,
                    }
                )
                # print(f"Debug: New leaf {leaf_number_counter} added on {current_date}")

            # Update hourly results for thermal time
            start_datetime = pd.Timestamp(current_date)
            end_datetime = start_datetime + pd.Timedelta(days=1)
            hourly_index = pd.date_range(start_datetime, end_datetime, freq="h", inclusive="left")

            result.loc[hourly_index, "cumulative_thermal_time_until_yesterday"] = (
                day_start_cumulative_tt
            )
            # Calculate hourly cumulative thermal time within the day
            # Check if 'Temperature' column exists before accessing (using correct capitalization)
            if "Temperature" not in self.df_hourly.columns:
                print(f"Error: 'Temperature' column not found in self.df_hourly.")
                print(f"Available columns: {self.df_hourly.columns.tolist()}")
                raise KeyError(
                    "Required column 'Temperature' is missing in the hourly climate data."
                )
            else:
                temp_col_name = "Temperature"  # Use the correct column name

            hourly_temps = self.df_hourly.loc[hourly_index, temp_col_name]
            hourly_growth = (hourly_temps - 10) / 24  # Distribute daily temp over 24h approx.
            # Ensure alignment before addition
            hourly_cumulative_growth = (
                hourly_growth.cumsum().reindex(hourly_index).ffill().fillna(0)
            )

            result.loc[hourly_index, "cumulative_thermal_time"] = (
                day_start_cumulative_tt + hourly_cumulative_growth
            )

            # Hourly loop for leaf removal check and setting hourly node/leaf counts
            for current_time in hourly_index:
                # --- Leaf Removal Check --- #
                removal_params = {
                    "config": self.leaf_removal_config,
                    "first_removal_done": self.first_removal_done,
                    "end_date_sim": self.end_date,
                    "leaf_removal_dates": current_leaf_removal_dates,
                    "result_df": result,  # Pass the result df for TT lookup
                    "gompertz_params": {
                        "a": self.a,
                        "b": self.b,
                        "c": self.c,
                    },  # Pass Gompertz params
                    "conversion_factor": self.conversion_factor,
                    "SLA": self.SLA,
                }

                try:
                    removed_this_step, updated_leaves_info, generated_dates_flag = (
                        process_leaf_removal(
                            current_time,
                            current_leaf_removal_dates,
                            leaves_info,
                            mode=self.leaf_removal_mode,
                            params=removal_params,
                        )
                    )
                    leaves_info = updated_leaves_info

                except Exception as e:
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(
                        f"[ERROR leaf_area] Exception in process_leaf_removal at {current_time}: {e}"
                    )
                    print(f"Current leaves_info length: {len(leaves_info)}")
                    import traceback

                    traceback.print_exc()  # 자세한 오류 정보 출력
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # 오류 발생 시, 이 시간 단계 처리를 건너뛰거나 기본값 설정 등 결정 필요
                    # 일단은 기존 leaves_info를 유지하고 진행하도록 함
                    removed_this_step = []
                    generated_dates_flag = None

                # Update based on removal results
                if removed_this_step:
                    removed_leaves_info.extend(removed_this_step)

                if generated_dates_flag is not None:
                    # Store the generated dates for the next call if needed
                    current_leaf_removal_dates = generated_dates_flag
                    # Update the flag indicating first removal is done
                    self.first_removal_done = True

                # Update hourly results AFTER potential leaf removal
                result.loc[current_time, "remaining_leaves"] = len(
                    leaves_info
                )  # Use length of (potentially updated) list
                result.loc[current_time, "total_nodes"] = nodes

            # --- End of Hourly Loop --- #
            cumulative_thermal_time = (
                day_start_cumulative_tt + daily_growing_temp
            )  # Update daily cumulative TT

        # Convert the detailed removal info to DataFrame at the end
        removed_leaves_info_df = pd.DataFrame(removed_leaves_info)
        # Convert the full leaf history to DataFrame
        all_leaves_history_df = pd.DataFrame(all_leaves_history)

        # Return the DataFrame of all leaves history instead of the final active leaves list
        # Return results
        # *** DEBUG LOG: Inspect removed_leaves_info before returning ***
        removed_df_for_log = pd.DataFrame(removed_leaves_info)
        if (
            not removed_df_for_log.empty
            and "Removed Leaf Area_m2" in removed_df_for_log.columns
            and (removed_df_for_log["Removed Leaf Area_m2"] > 1e-6).any()
        ):
            logger.info(
                "Leaf removal occurred. Content of removed_leaves_info just before returning from calculate_leaf_area:"
            )
            logger.info(
                removed_df_for_log[removed_df_for_log["Removed Leaf Area_m2"] > 1e-6].to_string()
            )
        elif not removed_df_for_log.empty:
            logger.info(
                "Leaf removal occurred but all Removed Leaf Area_m2 are zero or column missing."
            )
            logger.info(removed_df_for_log.head().to_string())
        else:
            logger.info("No leaf removal occurred in calculate_leaf_area.")
        # *** END DEBUG LOG ***

        self.max_leaf_no = (
            all_leaves_history_df["Leaf Number"].max() if not all_leaves_history_df.empty else 0
        )

        return result, all_leaves_history_df, pd.DataFrame(removed_leaves_info)

    def create_remaining_leaves_info(self, result_combined, leaves_info_df):
        """
        Create information about remaining leaves

        Args:
            result_combined (DataFrame): Combined results
            leaves_info_df (DataFrame): Leaf information

        Returns:
            list: Information about remaining leaves
        """
        remaining_leaves_info = []
        for i in range(len(result_combined)):
            row = result_combined.iloc[i]
            remaining_leaves_count = row["remaining_leaves"]
            cumulative_thermal_time_until_yesterday = row["cumulative_thermal_time_until_yesterday"]
            if not np.isnan(remaining_leaves_count) and remaining_leaves_count > 0:
                timestamp = row.name
                leaves_info_df["Date"] = pd.to_datetime(
                    leaves_info_df["Date"]
                )  # Ensure that 'Date' is datetime
                # Get the latest leaves based on appearance date
                # --- DEBUG LOG START ---
                filtered_leaves_before_tail = leaves_info_df[leaves_info_df["Date"] <= timestamp]
                # --- DEBUG LOG END ---
                current_leaves_info = (
                    filtered_leaves_before_tail.tail(  # Use the variable logged above
                        int(remaining_leaves_count)
                    )[["Leaf Number", "Thermal Time"]].copy()
                )
                # --- DEBUG LOG START ---
                if timestamp.hour == 0:  # Log only once per day to avoid excessive output
                    print(
                        f"[DEBUG create_remaining] Timestamp: {timestamp}, Remaining Count: {remaining_leaves_count}, Filtered Shape: {filtered_leaves_before_tail.shape}, Current Shape After Tail: {current_leaves_info.shape}"
                    )
                    if not current_leaves_info.empty:
                        print(
                            f"[DEBUG create_remaining] First Leaf No After Tail: {current_leaves_info.iloc[0]['Leaf Number']}, Last Leaf No After Tail: {current_leaves_info.iloc[-1]['Leaf Number']}"
                        )
                    elif not filtered_leaves_before_tail.empty:
                        print(
                            f"[DEBUG create_remaining] Filtered but empty after tail. Filtered First Leaf No: {filtered_leaves_before_tail.iloc[0]['Leaf Number']}, Filtered Last Leaf No: {filtered_leaves_before_tail.iloc[-1]['Leaf Number']}"
                        )
                    else:
                        print(f"[DEBUG create_remaining] Filtered is empty.")

                # --- DEBUG LOG END ---

                # Assign Rank: Newest leaf (last in tail) gets Rank 1, oldest remaining gets highest rank
                current_leaves_info["Rank"] = range(len(current_leaves_info), 0, -1)

                current_leaves_info["Timestamp"] = timestamp
                current_leaves_info["Leaf Area"] = current_leaves_info.apply(
                    lambda x: gompertz_growth(
                        cumulative_thermal_time_until_yesterday - x["Thermal Time"],
                        self.a,
                        self.b,
                        self.c,
                    ),
                    axis=1,
                )
                current_leaves_info["Leaf_Area_per_m2"] = (
                    current_leaves_info["Leaf Area"] * self.conversion_factor
                )
                remaining_leaves_info.append(current_leaves_info)
        return remaining_leaves_info

    def transform_remaining_leaves_info(self, remaining_leaves_info):
        """
        Transform remaining leaves info into a wide format DataFrame.

        Args:
            remaining_leaves_info (list): List of DataFrames with remaining leaf info per timestamp.

        Returns:
            list: Transformed data in list of lists format.
        """
        transformed_data = []
        # Concatenate all the small DataFrames first
        if not remaining_leaves_info:  # Handle empty input
            print(
                "[DEBUG transform_remaining_leaves_info] Input remaining_leaves_info is empty. Returning empty list."
            )
            return transformed_data

        all_remaining_leaves_df = pd.concat(remaining_leaves_info)

        if all_remaining_leaves_df.empty:
            print(
                "[DEBUG transform_remaining_leaves_info] Concatenated all_remaining_leaves_df is empty. Returning empty list."
            )
            return transformed_data

        # --- DEBUG LOGGING START ---
        try:
            min_ts_before = all_remaining_leaves_df["Timestamp"].min()
            max_ts_before = all_remaining_leaves_df["Timestamp"].max()
            print(
                f"[DEBUG transform_remaining_leaves_info] Before loop - Min Timestamp: {min_ts_before}, Max Timestamp: {max_ts_before}, Shape: {all_remaining_leaves_df.shape}"
            )
        except Exception as e:
            print(f"[ERROR transform_remaining_leaves_info] Error logging before loop: {e}")
        # --- DEBUG LOGGING END ---

        # Group by Timestamp
        for timestamp, group in all_remaining_leaves_df.groupby("Timestamp"):
            # Sort the group by Rank (ascending: Rank 1 first)
            # Rank 1 should correspond to the newest leaf based on create_remaining_leaves_info
            group_sorted = group.sort_values(by="Rank")

            row = [timestamp]
            # Iterate through the SORTED group
            for _, leaf_info in group_sorted.iterrows():
                row.extend(
                    [
                        leaf_info["Leaf Number"],
                        leaf_info["Thermal Time"],
                        leaf_info["Leaf Area"],
                        leaf_info["Leaf_Area_per_m2"],
                    ]
                )
            transformed_data.append(row)

        # --- DEBUG LOGGING START ---
        if transformed_data:
            try:
                first_ts_after = transformed_data[0][0]
                last_ts_after = transformed_data[-1][0]
                print(
                    f"[DEBUG transform_remaining_leaves_info] After loop - First Timestamp: {first_ts_after}, Last Timestamp: {last_ts_after}, Num Rows: {len(transformed_data)}"
                )
            except Exception as e:
                print(f"[ERROR transform_remaining_leaves_info] Error logging after loop: {e}")
        else:
            print("[DEBUG transform_remaining_leaves_info] After loop - transformed_data is empty.")
        # --- DEBUG LOGGING END ---

        return transformed_data

    def pad_transformed_data(self, transformed_data):
        """
        Pad transformed data with NaN values to ensure consistent dimensions

        Args:
            transformed_data (list): Transformed data

        Returns:
            tuple: (padded_data, max_length)
        """
        if not transformed_data:  # Handle empty input
            print(
                "[DEBUG pad_transformed_data] Input transformed_data is empty. Returning empty list and max_len 0."
            )
            return transformed_data, 0

        # --- DEBUG LOGGING START ---
        try:
            first_ts_before_pad = transformed_data[0][0]
            last_ts_before_pad = transformed_data[-1][0]
            print(
                f"[DEBUG pad_transformed_data] Before padding - First Timestamp: {first_ts_before_pad}, Last Timestamp: {last_ts_before_pad}, Num Rows: {len(transformed_data)}"
            )
        except Exception as e:
            print(f"[ERROR pad_transformed_data] Error logging before padding: {e}")
        # --- DEBUG LOGGING END ---

        max_len = 0
        for row in transformed_data:
            current_len = len(row) - 1  # Subtract 1 for the Timestamp value
            if current_len > max_len:
                max_len = current_len

        # Pad with NaN
        for row in transformed_data:
            # Add NaN values to make all rows the same length
            while len(row) < max_len + 1:  # +1 for Timestamp
                row.append(np.nan)

        # --- DEBUG LOGGING START ---
        try:
            first_ts_after_pad = transformed_data[0][0]
            last_ts_after_pad = transformed_data[-1][0]
            print(
                f"[DEBUG pad_transformed_data] After padding - First Timestamp: {first_ts_after_pad}, Last Timestamp: {last_ts_after_pad}, Max Len: {max_len}, Num Rows: {len(transformed_data)}"
            )
        except Exception as e:
            print(f"[ERROR pad_transformed_data] Error logging after padding: {e}")
        # --- DEBUG LOGGING END ---

        return transformed_data, max_len

    def create_columns(self, max_len):
        """
        Create column names for the final dataframe

        Args:
            max_len (int): Maximum number of leaves

        Returns:
            list: Column names
        """
        columns = ["Timestamp"]
        # Calculate how many leaf sets are in the data
        num_leaf_sets = max_len // 4
        for i in range(1, num_leaf_sets + 1):
            columns.extend(
                [f"Leaf_Rank_{i}", f"Thermal_Time_{i}", f"Leaf_Area_{i}", f"Leaf_Area_per_m2_{i}"]
            )
        return columns

    def get_removed_leaf_info(self):
        """Return the list of dictionaries containing removed leaf info."""
        return self.removed_leaves

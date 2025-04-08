"""
Photosynthesis calculation module for cucumber growth model
"""

import numpy as np
import pandas as pd
from src.configs.constants import (
    UNIVERSAL_GAS_CONSTANT,
    OXYGEN_CONCENTRATION,
    ACTIVATION_ENERGY,
    ENTROPY_TERMS,
    TEMP_DEPENDENCY,
    EXTINCTION_COEFFICIENT,
    RESPIRATION_COEFFICIENTS,
    TRANSPIRATION_PARAMS,
    CELSIUS_TO_KELVIN,
)
from src.configs.params import FvCB_PARAMETERS, QUANTUM_EFFICIENCY, STOMATAL_CONDUCTANCE, TIME_UNITS
import math


class FvCB_Calculator:
    """
    Farquhar-von Caemmerer-Berry (FvCB) photosynthesis model calculator
    Supports two modes for Ci calculation:
    - 'stomata': Based on a model linking stomatal conductance directly to A (default).
    - 'transpiration': Based on calculating transpiration and deriving total conductance (g_tc).
    """

    def __init__(self, df1, df2, plant_density, time_unit="hour", ci_calculation_mode="stomata"):
        """
        Initialize the FvCB Calculator

        Args:
            df1 (DataFrame): Environmental data (must include T, RH, CO2, PARi, 'Radiation intensity (W/m2)')
            df2 (DataFrame): Information about leaves
            plant_density (float): Plant density (plants per m^2), needed for Tr calculation.
            time_unit (str): Time unit for calculation ('minute' or 'hour')
            ci_calculation_mode (str): 'stomata' or 'transpiration'
        """
        self.df1 = df1.copy()  # Create a copy to avoid modifying original
        self.df2 = df2.copy()  # Create a copy to avoid modifying original
        self.final_leaves_info_df = df2.copy()  # Renamed to match original code
        self.plant_density = plant_density
        self.time_unit = time_unit
        self.ci_calculation_mode = ci_calculation_mode

        # Get relevant columns
        # Use the standardized column name 'Temperature' from climate.py processing
        self.T = self.df1["Temperature"]
        self.RH = self.df1["RH"]
        self.hs = self.df1["hs"]
        self.CO2 = self.df1["CO2"]
        self.Ci = self.df1["Ci"]
        # Use PARi which is prepared by the canopy module
        self.PARi = self.df1["PARi"]
        # I_Glob needed for Mode 2 Rad factor calculation
        self.I_Glob = self.df1["Radiation intensity (W/m2)"]

        self.Vcmax_25 = FvCB_PARAMETERS["top_leaves"]["Vcmax_25"]
        self.Jmax_25 = FvCB_PARAMETERS["top_leaves"]["Jmax_25"]
        self.Rd_25 = FvCB_PARAMETERS["top_leaves"]["Rd_25"]
        self.theta = FvCB_PARAMETERS["top_leaves"]["theta"]
        self.time_seconds = TIME_UNITS[time_unit]["seconds"]
        self.time_label = TIME_UNITS[time_unit]["label"]

        # Load constants for Mode 2 if needed
        if self.ci_calculation_mode == "transpiration":
            self.t_params = TRANSPIRATION_PARAMS

    # --- Helper methods for Transpiration-based Ci Calculation (Mode 2) ---
    def _calculate_psatvap(self, T):
        "Calculates saturation vapor pressure (kPa)."  # Eq from notebook
        return 0.61365 * math.exp((17.502 * T) / (240.97 + T))

    def _calculate_vpd(self, T, RH):
        "Calculates Vapor Pressure Deficit (kPa)."  # Standard formula
        psat = self._calculate_psatvap(T)
        actual_vp = psat * (RH / 100.0)
        return psat - actual_vp

    def _calculate_H2O_components(self, T, RH):
        "Calculates intermediate water vapor components using notebook formulas."
        psat = self._calculate_psatvap(T)
        air_press = self.t_params["air_press_kpa"]  # Typically 101.325

        # Wl: Water vapor concentration at saturation (mmol/mol) - Matches notebook
        Wl = (psat / air_press) * 1000

        # AD: Absolute humidity (g/m^3) - From notebook AD() formula
        # Check denominator to avoid division by zero
        ad_denominator = 100 - RH / air_press * psat
        if ad_denominator <= 1e-9:
            ad_val = 0  # Handle potential division by zero
        else:
            ad_val = (0.62198 * RH / air_press * psat) / ad_denominator * 1000

        # H2O: Actual water vapor concentration in air (mmol/mol) - From notebook H2O() formula
        # Uses the calculated ad_val
        H2O = (ad_val * 0.0224) / 18.02 * 1000  # Note: Constants 0.0224 and 18.02 are specific

        return Wl, H2O

    def _calculate_rad_factor(self, I_Glob):
        "Calculates the radiation factor used in Tr calculation (unit?)."
        # Formula from notebook Rad() function
        # Assuming I_Glob is in W/m2
        return self.t_params["rad_const1"] + self.t_params["rad_const2"] * (
            1 - math.exp(-self.t_params["rad_const3"] * I_Glob * 4.57)
        )  # Note: 4.57 eta_Glob_PAR_W used here

    def _calculate_vpd_factor(self, vpd):
        "Calculates the VPD factor used in Tr calculation (unit?)."
        # Formula from notebook VPD_() function
        return (
            self.t_params["vpd_factor_const1"]
            * math.exp(
                -self.t_params["vpd_factor_const2"]
                * ((vpd - self.t_params["vpd_factor_const3"]) ** 2)
            )
        ) / self.t_params["vpd_factor_const4"]

    def _calculate_tr(self, LAI, rad_factor, vpd_factor):
        "Calculates transpiration rate using notebook Tr() formula."
        # Core calculation part
        tr_intermediate = (
            self.t_params["tr_const1"]
            * (1 - math.exp(-self.t_params["tr_const2"] * LAI))
            * rad_factor
            + self.t_params["tr_const3"] * LAI * vpd_factor
        )

        # Apply the exact conversion factor from the notebook
        # Requires self.plant_density initialized in __init__
        # Note: constants 18 (mol mass H2O) and 600 (unit conversion?) are specific
        Tr = (((tr_intermediate * 1000) * self.plant_density * 100) / 18) / 600

        # Remove previous warning
        # print(
        #     f"[WARN] Transpiration (Tr) calculation needs unit validation. Intermediate: {tr_intermediate}"
        # )
        return Tr  # Return the value calculated with notebook conversion

    def _calculate_g_tw(self, Tr, Wl, H2O):
        "Calculates total water vapor conductance using notebook g_tw() formula."
        denominator = Wl - H2O
        if denominator <= 1e-9:  # Avoid division by zero or very small number
            return 0  # Or a very small conductance

        # Use the exact numerator from the notebook
        numerator = Tr * (1000 - 0.5 * (H2O + Wl))

        # Remove previous warning
        # print(
        #     f"[WARN] g_tw calculation using simplified Tr / (Wl-H2O). Original factor (1000 - 0.5*(H2O+Wl)) ignored due to unit uncertainty."
        # )
        return numerator / denominator  # Use the full notebook formula

    def _calculate_g_sw_transpiration(self, g_tw):
        "Calculates stomatal water conductance from total conductance (mol m-2 s-1)."
        # Formula from notebook g_sw() : 1 / ((1/g_tw) - (0.5 / g_bw))
        # Note: 0.5 might be related to parallel pathways or leaf sides.
        gb_term = 0.5 / self.t_params["g_bw"]
        if g_tw <= gb_term or g_tw <= 1e-9:  # Ensure 1/g_tw is valid and result is positive
            return 0  # Or very small positive value
        res_sw = (1.0 / g_tw) - gb_term
        if res_sw <= 1e-9:
            return 0
        return 1.0 / res_sw

    def _calculate_g_tc_transpiration(self, g_sw):
        "Calculates total CO2 conductance (mol m-2 s-1)."
        # Formula from notebook g_tc() : 1 / (1.6/g_sw + 1.37*0.56/g_bw)
        if g_sw <= 1e-9:
            return 0
        res_tc = (self.t_params["gtc_gs_factor"] / g_sw) + (
            self.t_params["gtc_gb_factor"] * self.t_params["gtc_gb_mult"] / self.t_params["g_bw"]
        )
        if res_tc <= 1e-9:
            return 0
        return 1.0 / res_tc

    # --- End Mode 2 Helpers ---

    def set_params_by_rank(self, rank):
        """
        Set photosynthesis parameters based on leaf rank

        Args:
            rank (int): Leaf rank
        """
        if rank <= 5:
            self.Vcmax_25 = FvCB_PARAMETERS["top_leaves"]["Vcmax_25"]
            self.Jmax_25 = FvCB_PARAMETERS["top_leaves"]["Jmax_25"]
            self.Rd_25 = FvCB_PARAMETERS["top_leaves"]["Rd_25"]
            self.theta = FvCB_PARAMETERS["top_leaves"]["theta"]
        elif rank <= 10:
            self.Vcmax_25 = FvCB_PARAMETERS["middle_leaves"]["Vcmax_25"]
            self.Jmax_25 = FvCB_PARAMETERS["middle_leaves"]["Jmax_25"]
            self.Rd_25 = FvCB_PARAMETERS["middle_leaves"]["Rd_25"]
            self.theta = FvCB_PARAMETERS["middle_leaves"]["theta"]
        else:
            self.Vcmax_25 = FvCB_PARAMETERS["lower_leaves"]["Vcmax_25"]
            self.Jmax_25 = FvCB_PARAMETERS["lower_leaves"]["Jmax_25"]
            self.Rd_25 = FvCB_PARAMETERS["lower_leaves"]["Rd_25"]
            self.theta = FvCB_PARAMETERS["lower_leaves"]["theta"]

    def calculate_PPF_REAL(self, i, rank):
        """
        Calculate the real PPF (Photosynthetic Photon Flux) received by a leaf
        considering light absorption by upper leaves using Beer-Lambert law.
        This version aligns with the Cucumber.py logic where attenuation
        is caused by leaves *above* the current rank.

        Args:
            i (int): Index in the dataframe
            rank (int): Rank of the leaf (1 = newest/top leaf, higher numbers = older/lower leaves)

        Returns:
            float: Real PPF received by the leaf
        """
        current_time = self.df1.index[i]
        PAR_incident = self.df1.iloc[i]["PARi"]

        # Sum LAI of leaves strictly *above* the current rank (ranks 1 to rank-1)
        SUM_RANK_LAI = 0
        if rank > 1:  # Only calculate attenuation if there are leaves above
            for j in range(1, rank):
                try:
                    # Revert to using timestamp index access as prepared in canopy.py
                    leaf_area_col = f"Leaf_Area_per_m2_{j}"
                    if (
                        current_time in self.final_leaves_info_df.index
                        and leaf_area_col in self.final_leaves_info_df.columns
                    ):
                        leaf_area = self.final_leaves_info_df.loc[current_time, leaf_area_col]

                        if not np.isnan(leaf_area):
                            SUM_RANK_LAI += leaf_area
                except (IndexError, KeyError):
                    # If a column for rank 'j' doesn't exist, skip it
                    continue

        # Apply Beer-Lambert law using the cumulative LAI of leaves *above*
        PPF_REAL = PAR_incident * np.exp(-EXTINCTION_COEFFICIENT * SUM_RANK_LAI)
        return PPF_REAL

    def calculate_A(self, i, rank):
        """
        Calculate photosynthesis rate A

        Args:
            i (int): Index in df1
            rank (int): Leaf rank

        Returns:
            tuple: (A, Rd) or (None, None) if calculation fails
        """
        T = self.T.iloc[i]
        Ci = self.Ci.iloc[i]
        PARi = self.calculate_PPF_REAL(i, rank)
        V_Ha = ACTIVATION_ENERGY["V_Ha"]
        R = UNIVERSAL_GAS_CONSTANT
        V_S = ENTROPY_TERMS["V_S"]
        V_Hd = ACTIVATION_ENERGY["V_Hd"]
        J_Ha = ACTIVATION_ENERGY["J_Ha"]
        J_S = ENTROPY_TERMS["J_S"]
        J_Hd = ACTIVATION_ENERGY["J_Hd"]
        O = OXYGEN_CONCENTRATION
        a = QUANTUM_EFFICIENCY
        theta = self.theta

        Vcmax_25 = self.Vcmax_25
        Jmax_25 = self.Jmax_25
        Rd_25 = self.Rd_25

        T_K = T + 273.15  # Convert Celsius to Kelvin
        gammastar = TEMP_DEPENDENCY["Gamma25"] * np.exp(37830 * (T_K - 298) / (298 * R * T_K))
        Kc = TEMP_DEPENDENCY["Kc25"] * np.exp(79430 * (T_K - 298) / (298 * R * T_K))
        Ko = TEMP_DEPENDENCY["Ko25"] * np.exp(36380 * (T_K - 298) / (298 * R * T_K))
        Rd = Rd_25 * 2 ** ((T - 25) / 10)

        Vc = (
            Vcmax_25
            * ((31 + (69 / (1 + np.exp(-0.005 * (PARi - 350))))) / 100)
            * np.exp(V_Ha * (T - 25) / ((25 + 237.15) * R * T_K))
            * (
                (1 + np.exp((V_S - V_Hd) / ((25 + 273.15) * R)))
                / (1 + np.exp((V_S - V_Hd) / (T_K * R)))
            )
        )
        A1 = (Vc * (Ci - gammastar) / (Ci + Kc * (1 + O / Ko))) - Rd

        Jmax = (
            Jmax_25
            * np.exp(J_Ha * (T - 25) / ((25 + 237.15) * R * T_K))
            * (
                (1 + np.exp((J_S - J_Hd) / ((25 + 273.15) * R)))
                / (1 + np.exp((J_S - J_Hd) / (T_K * R)))
            )
        )
        J = (a * PARi + Jmax - np.sqrt((a * PARi + Jmax) ** 2 - 4 * theta * a * PARi * Jmax)) / (
            2 * theta
        )
        A2 = (J * (Ci - gammastar) / (4 * Ci + 8 * gammastar)) - Rd

        if np.isnan(A1) and np.isnan(A2):
            print(f"NaN detected in both A1 and A2 at Row: {i}, Rank: {rank}")
            return None, None

        if A1 < A2:
            A = A1
        else:
            A = A2

        return A, Rd

    def calculate_gsc_gsw(self, i, A):
        """
        Calculate stomatal conductance based on Stomatal Conductance model (Mode 1).

        Args:
            i (int): Index in df1
            A (float): Photosynthesis rate

        Returns:
            tuple: (gsc, gsw) - conductance values
        """
        h = self.hs.iloc[i]
        CO2 = self.CO2.iloc[i]
        gsw = ((A * STOMATAL_CONDUCTANCE["slope"] * h) / CO2) + STOMATAL_CONDUCTANCE["intercept"]
        gsc = gsw / 1.6

        return gsc, gsw

    def update_Ci_and_recalculate(self, i, rank, g_tc_for_mode2=None):
        """
        Update internal CO2 concentration and recalculate photosynthesis
        based on the selected ci_calculation_mode.

        Args:
            i (int): Index in df1
            rank (int): Leaf rank
            g_tc_for_mode2 (float, optional): Pre-calculated total CO2 conductance for Mode 2.

        Returns:
            tuple: (A, gsc, gsw, Ci_new, Rd) or None if calculation fails
        """
        self.set_params_by_rank(rank)
        Ci_initial = self.Ci.iloc[i]
        iteration = 0  # Iteration count variable for debugging
        while True:
            A, Rd = self.calculate_A(i, rank)
            if A is None:
                return None  # return None if NaN is encountered

            if self.ci_calculation_mode == "stomata":
                # Mode 1: Calculate gsc from A using the stomatal model
                gsc, gsw = self.calculate_gsc_gsw(i, A)
                if gsc <= 1e-9:  # Avoid division by zero/negative
                    print(
                        f"[WARN] Mode 1: Non-positive gsc ({gsc:.2e}) calc. Row {i}, Rank {rank}. Setting Ci=CO2."
                    )
                    Ci_new = self.CO2.iloc[i]
                else:
                    Ci_new = self.CO2.iloc[i] - A / gsc
            elif self.ci_calculation_mode == "transpiration":
                # Mode 2: Use pre-calculated canopy g_tc
                g_tc = g_tc_for_mode2
                if g_tc is None or g_tc <= 1e-9:  # Ensure g_tc is valid
                    print(
                        f"[WARN] Mode 2: Invalid g_tc ({g_tc}) provided. Row {i}, Rank {rank}. Setting Ci=CO2."
                    )
                    Ci_new = self.CO2.iloc[i]
                    gsw = 0  # Cannot calculate gsw reliably
                    gsc = 0
                else:
                    Ci_new = self.CO2.iloc[i] - A / g_tc
                    # Estimate gsw/gsc based on calculated A for consistency in return value
                    # This is an approximation using the Mode 1 model
                    gsc, gsw = self.calculate_gsc_gsw(i, A)
            else:
                raise ValueError(f"Unknown ci_calculation_mode: {self.ci_calculation_mode}")

            if abs(Ci_new - Ci_initial) <= 0.001 or iteration > 100:
                if iteration > 100:
                    print(f"Max iterations reached for row {i}, Rank: {rank}")
                return A, gsc, gsw, Ci_new, Rd

            # Update the 'Ci' value directly in the original DataFrame (self.df1)
            # This ensures the updated Ci is used for the next rank calculation within the same timestamp
            self.df1.loc[self.df1.index[i], "Ci"] = Ci_new

            Ci_initial = Ci_new
            iteration += 1


def calculate_photosynthesis_dataframe(
    df_combined,
    final_leaves_info_df,
    plant_density,
    time_unit="hour",
    ci_calculation_mode="stomata",
):
    """
    Calculate photosynthesis rates for multiple timepoints from dataframe
    Implementation follows original Cucumber.py logic

    Args:
        df_combined (DataFrame): Combined dataframe with environmental data
        final_leaves_info_df (DataFrame): DataFrame with leaf information
        plant_density (float): Plant density (plants/m^2), needed for Tr calculation.
        time_unit (str): Time unit for calculations
        ci_calculation_mode (str): 'stomata' or 'transpiration' for Ci calculation method.

    Returns:
        tuple: (Rank photosynthesis DataFrame, Total photosynthesis DataFrame)
    """
    # Get time unit settings
    time_seconds = TIME_UNITS[time_unit]["seconds"]
    time_label = TIME_UNITS[time_unit]["label"]

    calculator = FvCB_Calculator(
        df_combined, final_leaves_info_df, plant_density, time_unit, ci_calculation_mode
    )
    results = []
    total_photosynthesis_by_time = []

    for i in range(len(df_combined)):
        timestamp = df_combined.index[i]
        total_gross_photosynthesis_amount = 0
        # Initialize Tr_value for this timestamp
        Tr_value = np.nan  # Initialize Tr_value outside the mode check

        # --- Pre-calculate canopy level values for Mode 2 --- #
        g_tc_for_mode2 = None
        Wl = np.nan  # Initialize Wl
        H2O = np.nan  # Initialize H2O
        if ci_calculation_mode == "transpiration":
            try:
                # 1. Get current environmental conditions
                current_T = df_combined.iloc[i]["Temperature"]
                current_RH = df_combined.iloc[i]["RH"]
                current_I_Glob = df_combined.iloc[i]["Radiation intensity (W/m2)"]
                current_LAI = df_combined.iloc[i]["LAI"]
                # hs is not directly used in these helpers, but checked for completeness
                current_hs = df_combined.iloc[i]["hs"]

                # Check for necessary columns and non-null values
                required_cols_mode2 = [
                    "Temperature",
                    "RH",
                    "Radiation intensity (W/m2)",
                    "LAI",
                    "hs",  # Keep hs check for now
                ]
                if not all(col in df_combined.columns for col in required_cols_mode2):
                    pass  # Skip calculations if columns missing
                elif df_combined.iloc[i][required_cols_mode2].isnull().any():
                    pass  # Skip calculations if values are null
                else:
                    # 2. Calculate intermediate factors
                    rad_factor = calculator._calculate_rad_factor(current_I_Glob)
                    vpd = calculator._calculate_vpd(current_T, current_RH)  # hs is not used here
                    vpd_factor = calculator._calculate_vpd_factor(vpd)
                    # Calculate Wl and H2O needed for g_tw
                    Wl, H2O = calculator._calculate_H2O_components(current_T, current_RH)

                    # 3. Calculate Tr and store it
                    Tr_value = calculator._calculate_tr(current_LAI, rad_factor, vpd_factor)

                    # 4. Calculate g_tc based on Tr, passing Wl and H2O to g_tw
                    g_tw = calculator._calculate_g_tw(Tr_value, Wl, H2O)  # Pass Wl and H2O
                    g_sw = calculator._calculate_g_sw_transpiration(g_tw)
                    g_tc_for_mode2 = calculator._calculate_g_tc_transpiration(g_sw)

            except KeyError as ke:
                print(
                    f"[ERROR] Missing key '{ke}' for Mode 2 calculation at {timestamp}. Check df_combined columns."
                )
            except Exception as e:
                print(f"[ERROR] Failed to calculate g_tc/Tr for Mode 2 at {timestamp}: {e}")
        # --- End Mode 2 Pre-calculation --- #

        remaining_leaves_count = df_combined.iloc[i]["remaining_leaves"]
        if np.isnan(remaining_leaves_count):
            continue

        for rank in range(1, int(remaining_leaves_count) + 1):
            try:
                # Follow exact same approach as original code for consistency
                # Revert to using timestamp index access
                leaf_area_col = f"Leaf_Area_per_m2_{rank}"
                if (
                    timestamp in final_leaves_info_df.index
                    and leaf_area_col in final_leaves_info_df.columns
                ):
                    leaf_area_per_m2 = final_leaves_info_df.loc[timestamp, leaf_area_col]
                else:
                    continue  # Skip if timestamp or column not found

                if np.isnan(leaf_area_per_m2):
                    continue
            except (IndexError, KeyError):
                continue

            # Pass pre-calculated g_tc if in Mode 2
            calculation_result = calculator.update_Ci_and_recalculate(i, rank, g_tc_for_mode2)
            if calculation_result is None:
                continue

            A, gsc, gsw, Ci_new, Rd = calculation_result
            PPF_REAL = calculator.calculate_PPF_REAL(i, rank)
            gross_A = A + Rd
            rank_gross_photosynthesis_rate = gross_A * leaf_area_per_m2
            rank_gross_photosynthesis_amount = (
                rank_gross_photosynthesis_rate * time_seconds
            )  # photosynthesis for the time period
            total_gross_photosynthesis_amount += rank_gross_photosynthesis_amount

            results.append(
                {
                    "Timestamp": timestamp,
                    "Rank": rank,
                    "gross_A": gross_A,
                    f"rank_gross_photosynthesis_rate/{time_label}": rank_gross_photosynthesis_rate,
                    f"rank_gross_photosynthesis_amount/{time_label}": rank_gross_photosynthesis_amount,
                    "PPF_REAL": PPF_REAL,
                }
            )

        # After iterating through leaves for the current timestamp
        # Use the dynamic key name expected by canopy.py
        time_results = {
            "Timestamp": timestamp,
            # Use the dynamic key based on time_unit label
            f"total_gross_photosynthesis_amount/{time_label}": total_gross_photosynthesis_amount,
            # Keep other relevant info
            "Average_Ci": Ci_new if "Ci_new" in locals() else np.nan,  # Ensure Ci_new exists
            "Average_gsc": gsc if "gsc" in locals() else np.nan,  # Ensure gsc exists
            "Tr_mol_m2_s": Tr_value,
        }
        total_photosynthesis_by_time.append(time_results)

    # Create DataFrame from the list of dictionaries
    total_photosynthesis_df = pd.DataFrame(total_photosynthesis_by_time)
    total_photosynthesis_df.set_index("Timestamp", inplace=True)

    return pd.DataFrame(results), total_photosynthesis_df


def calculate_rm(result_combined):
    """
    Calculate maintenance respiration rates

    Args:
        result_combined (DataFrame): Combined results data

    Returns:
        DataFrame: Maintenance respiration dataframe
    """
    # Create a copy of the result_combined DataFrame
    result_combined_copy = result_combined.copy()

    # Check for and convert date columns to proper datetime format
    date_columns = []
    for column in result_combined_copy.columns:
        if result_combined_copy[column].dtype == "object":
            # Check if the column contains date objects
            try:
                sample_value = result_combined_copy[column].iloc[0]
                if (
                    hasattr(sample_value, "year")
                    and hasattr(sample_value, "month")
                    and hasattr(sample_value, "day")
                ):
                    # Convert date columns to strings to prevent date arithmetic issues
                    result_combined_copy[column] = result_combined_copy[column].astype(str)
                    date_columns.append(column)
            except (IndexError, AttributeError, TypeError):
                pass

    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(result_combined_copy.index, pd.DatetimeIndex):
        if "Timestamp" in result_combined_copy.columns:
            # If there's a Timestamp column, set it as the index
            result_combined_copy = result_combined_copy.set_index("Timestamp")
        elif "DateTime" in result_combined_copy.columns:
            # If there's a DateTime column, set it as the index
            result_combined_copy = result_combined_copy.set_index("DateTime")
        else:
            # If no datetime column is found, try to convert the index to datetime
            try:
                result_combined_copy.index = pd.to_datetime(result_combined_copy.index)
            except Exception as e:
                raise ValueError(f"Unable to create a DatetimeIndex for resampling: {e}")

    # Use numeric_only=True to avoid issues with date columns
    daily_avg_temp_df = result_combined_copy.resample("D").mean(numeric_only=True)

    # Check if temperature column exists
    if "Temperature" in daily_avg_temp_df.columns:
        daily_avg_temp = daily_avg_temp_df["Temperature"]
    else:
        raise ValueError("Temperature column not found in the DataFrame")

    # Convert index to date format
    daily_avg_temp.index = daily_avg_temp.index.date

    # Calculate Rm_Vegetative and Rm_Fruit
    rm_vegetative = RESPIRATION_COEFFICIENTS["vegetative"] * (2 ** ((daily_avg_temp - 25) / 10))
    rm_fruit = RESPIRATION_COEFFICIENTS["fruit"] * (2 ** ((daily_avg_temp - 25) / 10))

    rm_df = pd.DataFrame(
        {
            "Date": daily_avg_temp.index,
            "Rm_Vegetative (CH2Og/g DM)": rm_vegetative.values,
            "Rm_Fruit (CH2Og/g DM)": rm_fruit.values,
        }
    )

    return rm_df

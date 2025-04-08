# ----
# Flowering and Fruit Set Module (Based on Research Paper)
# ----
# Description: Functions for calculating daily flower appearance rate (N),
#              potential fruit set based on source/sink ratio (NR),
#              and determining the actual number of fruits set per day.
#              The selection of which specific flowers set fruit (up to the calculated limit)
#              needs to be handled by the calling module using a sequential approach.
# Author: Your Name/AI Assistant (Adapted from research paper logic)
# Created: 2023-10-27 # Adjust date as needed
# Version: 2.0.0 # Major revision based on paper
# ----

import logging
import math

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Constants based on the paper for NR calculation
TEMP1_NR = 17.6  # °C
INTERCEPT1_NR = 0.5
SLOPE1_NR = 4.7

TEMP2_NR = 23.7  # °C
INTERCEPT2_NR = -0.4
SLOPE2_NR = 8.1

# --- Helper function for NR interpolation ---


def interpolate_nr_params(temperature: float) -> tuple[float, float]:
    """
    Linearly interpolates the intercept and slope for the NR calculation
    based on the given temperature and the two reference points from the paper.

    Args:
        temperature: The average daily temperature (°C).

    Returns:
        A tuple containing the interpolated (intercept, slope) for the NR formula.
        Performs linear interpolation/extrapolation.
    """
    if TEMP2_NR == TEMP1_NR:  # Avoid division by zero
        if temperature == TEMP1_NR:
            return (INTERCEPT1_NR, SLOPE1_NR)
        else:
            # Handle case where reference temps are the same but input is different
            # Could return average, raise error, or use one of the points
            logger.warning(
                f"NR reference temperatures are identical ({TEMP1_NR}°C), using params for this temp."
            )
            return (INTERCEPT1_NR, SLOPE1_NR)  # Or INTERCEPT2_NR, SLOPE2_NR

    # Calculate interpolation factor (alpha)
    alpha = (temperature - TEMP1_NR) / (TEMP2_NR - TEMP1_NR)

    # Interpolate intercept and slope
    interpolated_intercept = INTERCEPT1_NR + alpha * (INTERCEPT2_NR - INTERCEPT1_NR)
    interpolated_slope = SLOPE1_NR + alpha * (SLOPE2_NR - SLOPE1_NR)

    logger.debug(
        f"Interpolated NR params at {temperature}°C: intercept={interpolated_intercept:.3f}, slope={interpolated_slope:.3f}"
    )
    return interpolated_intercept, interpolated_slope


# --- Core Calculation Functions ---


def calculate_flower_appearance_rate(temperature: float, par: float) -> float:
    """
    Calculates the main stem flower appearance rate (N) based on temperature and PAR.
    Formula adapted from Challa & Van De Vooren (1980) as cited in the paper (Eq. 7).

    Args:
        temperature: Average daily air temperature (°C).
        par: Daily Photosynthetically Active Radiation at the top of the canopy (MJ m⁻² d⁻¹).
             Ensure units are correct before passing.

    Returns:
        The calculated flower appearance rate (N, flowers plant⁻¹ day⁻¹).

    Raises:
        ValueError: If PAR is negative.
    """
    if par < 0:
        logger.error(f"Invalid input: PAR cannot be negative, got {par}")
        raise ValueError("PAR cannot be negative.")

    # Formula: N = (-0.75 + 0.09 * T) * (1 - e^(-0.5 - 0.5 * R))
    temp_factor = -0.75 + 0.09 * temperature
    par_factor = 1.0 - math.exp(-0.5 - 0.5 * par)

    # Ensure rate is not negative (e.g., at very low temperatures)
    flower_rate_n = max(0.0, temp_factor * par_factor)

    logger.debug(
        f"Calculated flower appearance rate (N) at T={temperature}°C, PAR={par} MJ m⁻² d⁻¹: {flower_rate_n:.3f}"
    )
    return flower_rate_n


def calculate_potential_fruit_set(source_sink_ratio: float, temperature: float) -> float:
    """
    Calculates the potential number of non-aborting young fruits (NR)
    based on the source/sink ratio (SO/SI) and temperature.
    Uses linearly interpolated parameters based on the paper's findings (Eq. 8).

    Args:
        source_sink_ratio: The source/sink ratio (SO/SI) for the day.
                           The paper suggests using min(current SO/SI, 5-day average SO/SI).
                           This calculation must be done *before* calling this function.
        temperature: Average daily air temperature (°C).

    Returns:
        The potential number of fruits that can be set (NR, fruits plant⁻¹ day⁻¹).
        The value is capped at a minimum of 0.

    Raises:
        ValueError: If source_sink_ratio is negative.
    """
    if source_sink_ratio < 0:
        logger.error(
            f"Invalid input: Source/Sink ratio cannot be negative, got {source_sink_ratio}"
        )
        raise ValueError("Source/Sink ratio cannot be negative.")

    # Get interpolated parameters for the NR formula
    intercept, slope = interpolate_nr_params(temperature)

    # Formula: NR = intercept + slope * (SO/SI)
    potential_set_nr = intercept + slope * source_sink_ratio

    # Ensure the number of fruits is not negative
    potential_set_nr = max(0.0, potential_set_nr)

    logger.debug(
        f"Calculated potential fruit set (NR) at T={temperature}°C, SO/SI={source_sink_ratio:.3f}: {potential_set_nr:.3f}"
    )
    return potential_set_nr


def determine_actual_fruit_set(
    daily_flower_appearance_rate_n: float,
    daily_potential_fruit_set_nr: float,
    flower_emergence_remainder: float = 0.0,
) -> tuple[int, float]:
    """
    Determines the actual number of fruits set today based on the number of
    flowers appearing and the plant's capacity (potential fruit set).

    This function handles the fractional part of the flower appearance rate,
    carrying over any remainder to the next day.

    Args:
        daily_flower_appearance_rate_n: The calculated flower appearance rate (N) for the day.
        daily_potential_fruit_set_nr: The calculated potential fruit set (NR) for the day.
        flower_emergence_remainder: The fractional part of N carried over from the previous day.

    Returns:
        A tuple containing:
            - actual_fruits_set (int): The number of new fruits successfully set today.
            - next_day_remainder (float): The remaining fractional part of N to be carried
                                           over to the next day's calculation.
    """
    # Total potential flowers emerging today = Rate(N) + Remainder from yesterday
    total_potential_flowers = daily_flower_appearance_rate_n + flower_emergence_remainder

    # Number of whole flowers actually appearing today (integer part)
    num_flowers_appearing_today_k = math.floor(total_potential_flowers)

    # Calculate the remainder to carry over
    next_day_remainder = total_potential_flowers - num_flowers_appearing_today_k

    # Potential fruit set might be fractional, take floor as we set whole fruits
    max_settable_fruits = math.floor(daily_potential_fruit_set_nr)

    # Actual number of fruits set is the minimum of flowers appearing and potential set
    actual_fruits_set = min(num_flowers_appearing_today_k, max_settable_fruits)

    # Ensure non-negative result
    actual_fruits_set = max(0, actual_fruits_set)

    logger.debug(
        f"Fruit Set: N={daily_flower_appearance_rate_n:.3f}, NR={daily_potential_fruit_set_nr:.3f}, "
        f"RemainderIn={flower_emergence_remainder:.3f} -> "
        f"PotentialFlowers={total_potential_flowers:.3f}, Appeared(k)={num_flowers_appearing_today_k}, "
        f"MaxSettable={max_settable_fruits}, ActualSet={actual_fruits_set}, "
        f"RemainderOut={next_day_remainder:.3f}"
    )

    return actual_fruits_set, next_day_remainder


# --- Integration Note ---
# The calling module (e.g., canopy.py) is responsible for:
# 1. Calculating the daily Source/Sink ratio (SO/SI), considering the 5-day average rule.
# 2. Getting daily average Temperature and PAR (in MJ m⁻² d⁻¹).
# 3. Maintaining the `flower_emergence_remainder` state between days.
# 4. Calling these functions daily to get `actual_fruits_set`.
# 5. Managing the list of nodes that have flowered but not yet set fruit.
# 6. Applying the "Simple Sequential Approach": Assigning the `actual_fruits_set`
#    to the earliest nodes in the waiting list.
# 7. Tracking the individual weight of fruits on set nodes.

# Example Usage (Illustrative - requires external state management)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     # --- Mock daily inputs (These would come from the main simulation) ---
#     avg_temp_today = 20.0  # °C
#     par_today = 10.0      # MJ m⁻² d⁻¹
#     so_si_ratio_today = 0.8 # Example SO/SI
#     remainder_from_yesterday = 0.3 # Example remainder
#
#     # --- Daily Calculations ---
#     rate_n = calculate_flower_appearance_rate(avg_temp_today, par_today)
#     potential_nr = calculate_potential_fruit_set(so_si_ratio_today, avg_temp_today)
#     num_set_today, remainder_for_tomorrow = determine_actual_fruit_set(
#         rate_n, potential_nr, remainder_from_yesterday
#     )
#
#     print(f"\n--- Example Day ---")
#     print(f"Inputs: Temp={avg_temp_today}, PAR={par_today}, SO/SI={so_si_ratio_today}, RemainderIn={remainder_from_yesterday}")
#     print(f"Calculated: N={rate_n:.3f}, NR={potential_nr:.3f}")
#     print(f"Result: Fruits Set Today = {num_set_today}, Remainder for Tomorrow = {remainder_for_tomorrow:.3f}")
#
#     # --- The calling code would then manage the node list ---
#     # flowering_nodes_waiting = [5, 6, 7] # Example list of nodes flowered but not set
#     # fruits_to_assign = num_set_today
#     # nodes_set_today = []
#     # while fruits_to_assign > 0 and flowering_nodes_waiting:
#     #     node_to_set = flowering_nodes_waiting.pop(0) # Get earliest node
#     #     nodes_set_today.append(node_to_set)
#     #     # Initialize fruit weight tracking for node_to_set in canopy module
#     #     fruits_to_assign -= 1
#     # print(f"Nodes assigned fruit set today: {nodes_set_today}")
#     # print(f"Remaining waiting nodes: {flowering_nodes_waiting}")
#     # print(f"(State 'remainder_for_tomorrow' ({remainder_for_tomorrow:.3f}) must be stored for next day)")

"""
Model parameters for cucumber growth simulation
"""

# Default settings for the model
DEFAULT_SETTINGS = {
    "file_path": "data/input/climate_data.xlsx",  # Path with input environmental data
    "sheet_name": "Sheet1",  # Sheet name of input environmental data
    "start_date": "2021-02-23",
    "initial_nodes": 3,
    "threshold_before": 26.3,  # Thermal time threshold before split date
    "threshold_after": 15.6,  # Thermal time threshold after split date
    "split_date": "2021-03-26",
    "leaf_removal_dates_file": "data/input/leaf_removal_dates.xlsx",  # Path with leaf removal date information
    "leaf_removal_sheet_name": "Sheet1",  # Leaf removal sheet name
    "leaf_removal_mode": "file",  # Mode for leaf removal: 'file', 'interval', or 'threshold'
    "leaf_removal_interval_days": 7,  # Days between leaf removals when using interval mode
    "leaf_removal_time": "08:00:00",  # Time of day for leaf removal when using interval or threshold mode
    "leaf_removal_first_date": None,  # First date to start interval-based removal (YYYY-MM-DD)
    "leaf_removal_first_leaf_count": None,  # New parameter for first removal based on leaf count
    "output_file_path": "results/simulation_results.xlsx",  # Path to save result
    "SLA": 0.025,  # Specific Leaf Area (m^2 g^-1)
    "fruit_dw_file_path": "data/input/fruit_dw_data.xlsx",  # Path with daily harvested fruit weight information
    "fruit_dw_sheet_name": "2",
    "partitioning_vegetative_before": 1.0,  # Partitioning to vegetative before split date
    "partitioning_fruit_before": 0.0,  # Partitioning to fruit before split date
    "initial_remaining_vegetative_dw": 1.16,  # Initial vegetative dry weight (g m^-2)
    "plant_density_per_m2": 1.72,  # Plant density (plants m^-2)
    "time_unit": "hour",  # Time unit for photosynthesis calculation ('minute' or 'hour')
    "target_fruit_weight_per_node": 6.0,  # Target fruit dry weight for harvest (g/fruit)
    # Output storage settings
    "storage_format": "excel",  # Storage format: 'excel', 'csv', 'hdf5'
    "output_dir": "results",  # Directory to save results
    "save_hourly_data": True,  # Whether to save hourly data
    "save_daily_data": True,  # Whether to save daily data
    "save_leaf_data": True,  # Whether to save leaf-related data
    "compress_output": False,  # Whether to compress output files
    "create_summary": True,  # Whether to create a summary file
    "split_hourly_files": False,  # Whether to split hourly data into separate files (to prevent large files)
}

# Time units conversion factors
TIME_UNITS = {"minute": {"seconds": 60, "label": "min"}, "hour": {"seconds": 3600, "label": "hr"}}

# Storage format configurations
STORAGE_FORMATS = {
    "excel": {
        "extension": ".xlsx",
        "description": "Microsoft Excel file format",
        "single_file": True,
    },
    "csv": {
        "extension": ".csv",
        "description": "Comma-separated values text file",
        "single_file": False,
    },
    "hdf5": {
        "extension": ".h5",
        "description": "Hierarchical Data Format version 5",
        "single_file": True,
    },
}

# Output data categories
OUTPUT_CATEGORIES = {
    "hourly": [
        "hourly_data",
        "rank_photosynthesis_hourly",
        "total_photosynthesis_hourly",
        "final_leaves_hourly",
    ],
    "daily": ["daily_summary"],
    "leaf": ["leaves_info", "removed_leaves_info", "Node_Final_States"],
    "fruit": ["fruit_summary"],
    "summary": [],
}

# Photosynthesis model parameters - Leaf rank dependent
FvCB_PARAMETERS = {
    "top_leaves": {  # Ranks 1-5
        "Vcmax_25": 77.91,  # Updated to original value
        "Jmax_25": 132.45,  # Updated to original value
        "Rd_25": 0.63,  # Updated to original value
        "theta": 0.71,  # Updated to original value
    },
    "middle_leaves": {  # Ranks 6-10
        "Vcmax_25": 74.35,  # Updated to original value
        "Jmax_25": 126.40,  # Updated to original value
        "Rd_25": 0.35,
        "theta": 0.76,
    },
    "lower_leaves": {  # Ranks > 10
        "Vcmax_25": 63.82,
        "Jmax_25": 108.50,
        "Rd_25": 0.56,
        "theta": 0.88,
    },
}

# Gompertz growth parameters for leaf area development
GOMPERTZ_PARAMETERS = {
    "a": 582.06,  # Maximum value (asymptote)
    "b": 45.33,  # Displacement along x-axis
    "c": 45.31,  # Growth rate parameter
}

# Photosynthesis quantum efficiency parameter
QUANTUM_EFFICIENCY = 0.3  # Quantum efficiency of electron transport (mol e- mol^-1 photon)

# Stomatal conductance model parameters
STOMATAL_CONDUCTANCE = {
    "slope": 8.376,  # Slope of Ball-Berry stomatal conductance model
    "intercept": 0.045,  # Intercept of Ball-Berry stomatal conductance model
}

# Leaf removal management parameters
LEAF_REMOVAL = {
    "target_leaves": 5,  # Number of leaves to remove each time (default removal strategy)
    "target_remaining_leaves": 15,  # Number of leaves to remain after removal (alternate strategy)
    "remove_oldest_first": True,  # If True, remove oldest leaves first; if False, remove newest leaves
    "interval": {
        "days": 7,  # Days between removals
        "first_date": None,  # First date for removal
        "first_leaf_count": 20,  # Number of leaves at which to start first removal (Set to 20)
        "time": "08:00:00",  # Time of day for removal
    },
    "threshold": {
        "max_leaves": 20,  # Max number of leaves before removal
        "leaves_to_remove": 5,  # Number of leaves to remove when threshold is exceeded
    },
}

# Temperature dependency for maintenance respiration
TEMP_MAINT_RESP_COEFFICIENT = {"Q10": 2.0}  # Q10 value for maintenance respiration

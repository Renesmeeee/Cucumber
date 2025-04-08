# Cucumber Growth Model

A modular growth model for simulating cucumber plant growth dynamics, photosynthesis, and dry matter partitioning.

## Project Structure

```
cucumber_growth_model/
├── README.md               # Project description and quick start guide
├── pyproject.toml          # Project metadata and dependencies
├── main.py                 # Main entry point for running simulations
├── src/                    # Source code
│   ├── configs/            # Model parameters and constants 
│   │   ├── __init__.py
│   │   └── settings.py     # Simulation settings
│   ├── system/             # System definitions
│   │   ├── __init__.py
│   │   └── canopy.py       # Canopy system definition
│   ├── processes/          # Model processes
│   │   ├── __init__.py
│   │   ├── physiology/     # Physiological processes
│   │   │   ├── __init__.py
│   │   │   └── photosynthesis.py # Photosynthesis calculations
│   │   └── morphology/     # Morphological processes
│   │       ├── __init__.py
│   │       └── leaf_area.py  # Leaf area calculations
│   ├── environment/        # Environmental condition handling
│   │   ├── __init__.py
│   │   └── climate.py      # Climate data management
│   ├── management/         # Crop management functions
│   │   ├── __init__.py
│   │   └── leaf_removal.py # Leaf removal management
│   └── utils/
│       ├── __init__.py
│       └── function.py     # Common functions
└── tests/                  # Test code
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages: pandas, numpy, matplotlib, openpyxl

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cucumber_growth_model.git
cd cucumber_growth_model

# Install dependencies
pip install -e .
```

### Running the Simulation

1. Update the file paths in `main.py` to point to your input data files
2. Run the simulation:

```bash
python main.py
```

## Input Files Required

1. Environmental data file (Excel format) with columns:
   - DateTime: Timestamp
   - Air temperature: Temperature in °C
   - PPF: Photosynthetic Photon Flux
   - CO2: CO2 concentration
   - hs: Relative humidity
   - Ci: Initial internal CO2 concentration

2. Leaf removal dates file (Excel format) with columns:
   - Date: Date of leaf removal
   - Time: Time of leaf removal

3. Fruit dry weight file (Excel format) with columns:
   - Date: Date of harvest
   - Fruit DW/m^2: Harvested fruit dry weight per m²

## Output

The simulation produces an Excel file with the following sheets:
- Results: Hourly simulation results
- Leaf_Info: Information about each leaf
- Removed_Leaf_Info: Information about removed leaves
- Remaining_Leaves_Info: Information about remaining leaves
- Rank_Photosynthesis_Rates_GROSS: Photosynthesis rates by leaf rank
- Total_Photosynthesis_GROSS: Hourly total photosynthesis
- Daily_Photosynthesis_GROSS: Daily photosynthesis summary

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
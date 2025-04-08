"""
Physical and biological constants for cucumber growth model
"""

# Physical constants for photosynthesis
UNIVERSAL_GAS_CONSTANT = 8.314  # J/mol/K or Pa m^3/mol/K
STEFAN_BOLTZMANN = 5.67e-8  # W/m^2/K^4
CELSIUS_TO_KELVIN = 273.15  # Offset to convert Celsius to Kelvin
OXYGEN_CONCENTRATION = 210  # mmol/mol or 21%

# Enzyme kinetics parameters
ACTIVATION_ENERGY = {
    "V_Ha": 91185,  # J mol^-1, Activation energy for Vcmax
    "V_Hd": 202900,  # J mol^-1, Deactivation energy for Vcmax
    "J_Ha": 79500,  # J mol^-1, Activation energy for Jmax
    "J_Hd": 201000,  # J mol^-1, Deactivation energy for Jmax
}

ENTROPY_TERMS = {
    "V_S": 650,  # J mol^-1 K^-1, Entropy term for Vcmax
    "J_S": 650,  # J mol^-1 K^-1, Entropy term for Jmax
}

# Temperature-dependent parameters at 25°C
TEMP_DEPENDENCY = {
    "Kc25": 404.9,  # μmol mol^-1, Michaelis-Menten constant for CO2 at 25°C
    "Ko25": 278.4,  # mmol mol^-1, Michaelis-Menten constant for O2 at 25°C
    "Gamma25": 42.75,  # μmol mol^-1, CO2 compensation point at 25°C
}

# Energy requirements
GROWTH_EFFICIENCY = 1.45  # g DM / g CH2O, Growth efficiency coefficient

# Maintenance respiration coefficients
RESPIRATION_COEFFICIENTS = {
    "vegetative": 0.033,  # CH2O g g^-1 DM day^-1
    "fruit": 0.015,  # CH2O g g^-1 DM day^-1
}

# Light absorption coefficient
EXTINCTION_COEFFICIENT = 0.8  # Canopy light extinction coefficient

# Molecular conversion factors
MOL_MASS_CO2 = 44.01  # g mol^-1
MOL_MASS_CH2O = 30.031  # g mol^-1

# === Constants for Transpiration-based Ci calculation (Mode 2) ===
TRANSPIRATION_PARAMS = {
    "g_bw": 10.0,  # Boundary layer conductance (mol m-2 s-1) - Value from notebook comment. Confirm unit/value.
    "tr_const1": 0.616,  # Constant in Tr calculation
    "tr_const2": 0.86,  # Constant in Tr calculation (exponent for LAI)
    "tr_const3": 0.189,  # Constant in Tr calculation (factor for VPD)
    "rad_const1": 0.0005,  # Constant in Rad factor calculation
    "rad_const2": 0.0133,  # Constant in Rad factor calculation
    "rad_const3": 0.0007,  # Constant in Rad factor calculation
    "vpd_factor_const1": 0.0083,  # Constant in VPD factor calculation
    "vpd_factor_const2": 0.5,  # Constant in VPD factor calculation
    "vpd_factor_const3": 3.7086,  # Constant in VPD factor calculation
    "vpd_factor_const4": 1.8014,  # Constant in VPD factor calculation
    "gtc_gs_factor": 1.6,  # Factor for g_tc calculation (gsw to gsc)
    "gtc_gb_factor": 1.37,  # Factor for g_tc calculation (gbw related)
    "gtc_gb_mult": 0.56,  # Multiplier for gbw in g_tc calculation
    "water_mol_mass": 18.015,  # Molar mass of water (g/mol)
    "air_press_kpa": 101.325,  # Standard air pressure (kPa)
    # Note: UNIVERSAL_GAS_CONSTANT and CELSIUS_TO_KELVIN might be defined elsewhere already
    # "gas_const_r": 8.314,
    # "celsius_to_kelvin": 273.15
}
# === End Mode 2 Constants ===

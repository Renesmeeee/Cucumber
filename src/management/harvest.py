"""
Harvest management module for cucumber growth model
"""

import pandas as pd

def process_harvest_data(file_path, sheet_name='2'):
    """
    Process harvest data from file
    
    Args:
        file_path (str): Path to harvest data file
        sheet_name (str): Sheet name in harvest data file
        
    Returns:
        pandas.DataFrame: Processed harvest data indexed by date
    """
    try:
        # Read data from Excel file
        harvest_data = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Convert date column to datetime and set as index
        if 'Date' in harvest_data.columns:
            harvest_data['Date'] = pd.to_datetime(harvest_data['Date'])
            harvest_data.set_index('Date', inplace=True)
        
        return harvest_data
    except Exception as e:
        print(f"Error processing harvest data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def add_fruit_dw_column(climate_data, fruit_dw_data):
    """
    Add fruit dry weight column to climate data
    
    Args:
        climate_data (pandas.DataFrame): Hourly climate data
        fruit_dw_data (pandas.DataFrame): Daily fruit dry weight data
        
    Returns:
        pandas.DataFrame: Climate data with fruit dry weight column added
    """
    # Copy the climate data to avoid modifying the original
    result = climate_data.copy()
    
    # Initialize fruit dry weight column with zeros
    result['Fruit_DW'] = 0.0
    
    # If no fruit dry weight data, return the original data with zeros
    if fruit_dw_data is None or fruit_dw_data.empty:
        return result
    
    # For each day in the fruit dry weight data
    for date, row in fruit_dw_data.iterrows():
        # Get the fruit dry weight value
        if 'Fruit_DW' in row:
            dw_value = row['Fruit_DW']
        else:
            # If column name is different, use the first column
            dw_value = row.iloc[0]
        
        # Find entries in climate data for this date and set the fruit dry weight
        date_str = date.strftime('%Y-%m-%d')
        mask = result.index.strftime('%Y-%m-%d') == date_str
        result.loc[mask, 'Fruit_DW'] = dw_value
    
    return result 
def make_design_table(factor_level_dict):
    """
    Create a full factorial design from a dictionary of factors and their levels.
    
    Parameters:
    factor_level_dict (dict): A dictionary where keys are factor names and values are lists of factor levels.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the full factorial design.
    """
    import pandas as pd
    import itertools
    
    # Get all combinations of factor levels
    factor_names = list(factor_level_dict.keys())
    level_combinations = list(itertools.product(*factor_level_dict.values()))
    
    # Create a DataFrame from the combinations
    df = pd.DataFrame(level_combinations, columns=factor_names)
    
    # Important variables
    df["RUN"] = df.index + 1
    df["Done"] = 0
    df["Failure"] = 0

    # statespace_size is an integer setting. It is only a column of the design when it is a factor:
    # the settings which are not factors are added as columns by the caller afterwards
    if "statespace_size" in df.columns:
        df["statespace_size"] = df["statespace_size"].astype(int)

    return df
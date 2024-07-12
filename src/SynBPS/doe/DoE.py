
"""
GENR the experiments:
"""

def fix_label_values(df, run_settings, variables):
    import pandas as pd
    if 'Name_fix' not in df:
        df["Name_fix"] = 0
        for variable in variables:
            for run in df.index:
                idx = int(df[variable].loc[run])
                value = run_settings[variable][idx]
                df[variable].loc[run] = value

        df["Name_fix"] = 1
    
    #df = df.drop("Name_fix")
    return df

def fullfact_corrected(levels):
    import numpy as np
    import pandas as pd
    """
    Create a general full-factorial design
    
    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor
    
    Example
    -------
    ::
    
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
               
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H

def construct_df(x,r):
    import numpy as np
    import pandas as pd
    df=pd.DataFrame(data=x,dtype='float32')
    for i in df.index:
        for j in range(len(list(df.iloc[i]))):
            df.iloc[i][j]=r[j][int(df.iloc[i][j])]
    return df

def build_full_fact(factor_level_ranges):
    import numpy as np
    import pandas as pd
    """
    Builds a full factorial design dataframe from a dictionary of factor/level ranges
    Example of the process variable dictionary:
    {'Pressure':[50,60,70],'Temperature':[290, 320, 350],'Flow rate':[0.9,1.0]}
    """
    
    factor_lvl_count=[]
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lvl_count.append(len(factor_level_ranges[key]))
        factor_lists.append(factor_level_ranges[key])
    
    x = fullfact_corrected(factor_lvl_count)
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

def make_design_table(run_settings, file="design_table.csv"):
    """
    Creates an experiment table and stores it as a CSV
    """

    # Generate a full factorial:
    df=build_full_fact(run_settings)#[0:2]

    # Recode the string factor levels (recoding from natural number to string)
    df = fix_label_values(df, run_settings, variables = ["process_entropy",
                                                        "process_type",
                                                        "Deterministic_offset_W",
                                                        "first_state_model"])
    df = df.drop("Name_fix",axis=1)

    # Important variables
    df["RUN"] = df.index + 1
    df["Done"] = 0
    df["Failure"] = 0

    #change types
    df.statespace_size = df.statespace_size.astype(int)

    # save the design table 
    print("saved to",file)
    df.to_csv(file, index=False)

    return df
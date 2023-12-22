# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:21:24 2021

@author: Mike
"""

def GenerateInitialProb(D=["a","b"], p0_type="regular"):
    import numpy as np
    import pandas as pd
    
    if p0_type == "min_entropy":
        # Example P0 is one-hot
    
        P0 = np.zeros(len(D))
        P0[np.random.randint(0,len(D),1)[0]] = 1
        P0 = P0.tolist()

    if p0_type != "min_entropy":
        
        P0 = []
    
        for d in D:
            #Draw from uniform dist
            x_d = np.random.uniform(0,1,1)[0]
            #print(x_d)
            #Append the value to the vector
            P0.append(x_d)
        
        #Add the p(absorbtion)=0 to P0
        #P0.append(0)
        
        #Normalize
        S_sum = np.sum(P0)
        P0 = P0/S_sum
        
        #Make dataframe
        #P_0_df = pd.DataFrame(P0).T
        #P_0_df.columns = D
        
    return P0#P_0_df

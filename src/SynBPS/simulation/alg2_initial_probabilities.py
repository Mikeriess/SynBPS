# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:21:24 2021

@author: Mike
"""

def GenerateInitialProb(D=["a","b"], p0_type="regular"):
    import numpy as np
    import pandas as pd
    
    # Min entropy is one-hot
    if p0_type == "min_entropy":    
        P0 = np.zeros(len(D))
        P0[np.random.randint(0,len(D),1)[0]] = 1
        P0 = P0.tolist()
        
    """
    # Med entropy is random initial prob
    if p0_type == "med_entropy":
        
        P0 = []
    
        for d in D:
            #Draw from uniform dist
            x_d = np.random.uniform(0,1,1)[0]

            #Append the value to the vector
            P0.append(x_d)
    """

    # Med entropy is also one-hot; as most processes start with the same activity
    if p0_type == "med_entropy":    
        P0 = np.zeros(len(D))
        P0[np.random.randint(0,len(D),1)[0]] = 1
        P0 = P0.tolist()
        
    
    # Max entropy is equally likely
    if p0_type == "max_entropy":
        
        P0 = np.ones(len(D)).tolist()

    #Normalize
    S_sum = np.sum(P0)
    P0 = P0/S_sum
        
    return P0

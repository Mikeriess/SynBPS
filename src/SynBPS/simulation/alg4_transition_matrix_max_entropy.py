# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:24:43 2021

@author: Mike
"""

"""
Generation of transition matrix: Maximum entropy
"""


def Generate_transition_matrix_max_ent(D = ["a","b","c"]):
    import numpy as np
    import pandas as pd
    
    # Example P0 is one-hot
    
    P = []
    
    D_orig = D.copy()
    D_orig.append("END")
    
    
    for i in D_orig:
        #print(i)
        
        #draw a vector of ones
        R = np.ones((len(D_orig)))
        
        #normalize into probability range
        R = R/np.sum(R)
                
        #make sure no journeys are made from absorption state
        if i == D_orig[len(D_orig)-1]:
            R = np.zeros(len(D_orig))
            R[len(R)-1] = 1
        
        #append the vector R to P, which will be the matrix
        #append L as row of P
        P = np.hstack((P,R))
        
    #generate a dataframe with labels for the "to" states
    P_df = pd.DataFrame(np.reshape(P, newshape=(len(D_orig),len(D_orig))))
    P_df.columns = D_orig
    return P_df


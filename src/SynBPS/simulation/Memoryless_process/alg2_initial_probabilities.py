# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:21:24 2021

@author: Mike
"""

def GenerateInitialProb(D=["a","b"], p0_type="regular", seed_value=1337, rng=None):
    """
    Initial probabilities P0 over D (alg. 2): one-hot for min_entropy, random otherwise.

    Parameters
    ----------
    D : list of states, without the absorption state
    p0_type : min_entropy, med_entropy or max_entropy
    seed_value : seed of the stream memoryless_tables, used when rng is not given
    rng : numpy random generator, the stream of the tables of the memoryless process
    """
    import numpy as np
    from SynBPS.simulation.simulation_helpers import make_rng

    if rng is None:
        rng = make_rng(seed_value, "memoryless_tables")
    
    if p0_type == "min_entropy":
        # Example P0 is one-hot
    
        P0 = np.zeros(len(D))
        P0[rng.integers(0,len(D))] = 1
        P0 = P0.tolist()

    if p0_type != "min_entropy":
        
        P0 = []
    
        for d in D:
            #Draw from uniform dist
            x_d = rng.uniform(0,1)
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

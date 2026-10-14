# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 21:46:29 2021

@author: Mike
"""

"""
Generation of transition matrix: Minimum entropy
"""


def Generate_transition_matrix_min_ent(D = ["a","b","c","d","e"], P0=[], seed_value=1337, rng=None):
    """
    Transition matrix of one deterministic trace through every state (alg. 3).

    Parameters
    ----------
    D : list of states, without the absorption state
    P0 : one-hot initial probabilities, from alg. 2 with p0_type min_entropy
    seed_value : seed of the stream memoryless_tables, used when rng is not given
    rng : numpy random generator, the stream of the tables of the memoryless process
    """
    import numpy as np
    import pandas as pd
    from SynBPS.simulation.simulation_helpers import make_rng

    if rng is None:
        rng = make_rng(seed_value, "memoryless_tables")
    
 
    #absorption state is added to P0
    P0_abs = P0.copy()
    P0_abs.append(0)
    
    #absorption state is added to D
    D_abs = D.copy()
    D_abs.append("END")
    
    #get the starting initial state
    initial_state = P0_abs.index(1)
    initial_state = D_abs[initial_state]
    
    #remove initial state from set of states to sample from
    D_free = D.copy()
    D_free.remove(initial_state)
    
    # intial transition matrix
    P = np.zeros((len(D_abs),len(D_abs)))
    
    ##############################
    ##### STEP1 generate a trace
    
    #draw the trace with uniform probabilities
    trace = rng.choice(D_free, size=len(D_free), replace=False)
    trace = trace.tolist()
    
    #add the transient state in the end
    trace.append("END")
    
    # add the initial state to the beginning of the trace
    trace.insert(0,initial_state)
    #print(trace)
    
        #loop over each state in the full trace
    for i in list(range(0,len(trace))):
        
        #if not the last state (absorption)
        if i < len(trace)-1:
            state = trace[i]
            row = D_abs.index(state)
            
            nextstate = trace[i+1]
            col = D_abs.index(nextstate)
            
            #generate vector
            R = np.zeros(len(D_abs))
            R[col] = 1
            
            #print("from",state,"to",nextstate)
        #if the last state, set transition to self
        else:
            row = len(D_abs)-1
            #generate vector
            R = np.zeros(len(D_abs))
            #set last col to 1 for absorbtion state
            R[len(D_abs)-1] = 1
            
        # overwrite row in intial matrix P
        P[row,:] = R
    
        
    #generate a dataframe with labels for the "to" states
    P_df = pd.DataFrame(np.reshape(P, (len(D_abs),len(D_abs))))
    P_df.columns = D_abs
    
    #print("Min_Entropy: P0")
    #print(P0_abs)
    #print("Min_Entropy: Transition matrix")
    #print(P_df)

    return P_df#, P0


#P_df, P0 = Generate_transition_matrix_and_p0_min_ent(D = ["a","b","c","d","e"])


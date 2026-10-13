# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:35:33 2021

@author: Mike
"""


def States_not_reaching_absorption(P):
    """
    Indexes of the states from which the absorption state cannot be reached.

    Parameters
    ----------
    P : square transition matrix (numpy array), where the last row and column is the absorption state

    Returns
    -------
    unreachable : list of row indexes with no path (of transitions with positive probability) to the absorption state

    """
    import numpy as np

    n = len(P)

    #states which can reach the absorption state: the absorption state itself, and every state with a
    #positive probability of a transition into the set, until no more states are added
    reaching = {n-1}
    added = True

    while added:
        added = False
        for i in range(0,n):
            if i not in reaching and np.any(P[i, list(reaching)] > 0):
                reaching.add(i)
                added = True

    unreachable = [i for i in range(0,n) if i not in reaching]

    return unreachable


def Generate_transition_matrix_med_ent(D = ["a","b","c","d","e"],
                                       n_tranitions=3,
                                       kappa = 2,
                                       seed_value=1337):
    """
    Generation of transition matrix: Medium entropy (alg. 5)

    Parameters
    ----------
    D : list of states, such as D=["a","b","c","d","e"], without the absorption state
    
    n_tranitions : number of possible transitions from each state, drawn from D and the absorption state
    
    kappa : minimum of the total probability of a transition to the absorption state, summed over all states
        (the absorption state itself contributes 1). Random transitions to the absorption state are added
        until the sum is reached. At most 1 + len(D)/2, as the loop overwrites the absorption probability of
        a random state instead of adding to it, and becomes very slow when the states need more than half of
        their probability mass for absorption on average
    
    seed_value : not used, the global numpy stream is seeded by the caller

    Returns
    -------
    P : transition matrix as a pandas dataframe, with the absorption state END as the last row and column

    """
    import numpy as np

    #error handling
    if len(D) < n_tranitions:
        raise ValueError("med_ent_n_transitions cannot be larger than the statespace size. Change med_ent_n_transitions or statespace_size.")

    if n_tranitions < 2:
        raise ValueError("med_ent_n_transitions must be 2 or larger, as a single transition leads to an endless search. Change med_ent_n_transitions.")
    
    if kappa > 1 + len(D)/2:
        raise ValueError("med_ent_kappa must be at most 1 + statespace_size/2, as the search for the absorption probabilities becomes very slow above it. Change med_ent_kappa.")
    
    ####################################
    
    import numpy as np
    import pandas as pd
    
    D_orig = D.copy()
    D_orig.append("END")
            
    P = []
    #for every state in D
    for d in D_orig:
        #initialize L as a zero vector of length D
        L = np.zeros(len(D_orig))
        
        #draw n states from D, without replacement
        s = np.random.choice(D_orig, size=n_tranitions, replace=False)
                    
        #get indexes of states in s
        idx=[]
        for i in s:
            idx.append(D_orig.index(i))
            
        #draw probabilities for each state s
        #and replace that index of L with x_i
        for i in idx:
            x_i = np.random.uniform(0,1)
            L[i] = x_i
        
        #normalize to probability space
        L = L/np.sum(L)
        
        #Test1: make sure no journeys are made from absorption state
        if d == D_orig[len(D_orig)-1]:
            L = np.zeros(len(D_orig))
            L[len(L)-1] = 1
            
        
        #append L as row of P
        P = np.hstack((P,L))
        
    ####################################
    #convert P to a D x D matrix
    P = np.reshape(P,(len(D_orig),len(D_orig)))
    
    #Test2: Make sure _something_ leads to the absorption state
    
    # the last column (absorption probabilities) should sum to at least kappa. The absorption state contributes 1
    #rows, columns
    colsum = np.sum(P[:,len(D_orig)-1]) 
    
    while colsum < kappa:
        #get the last column
        lastcol = P[:,len(D_orig)-1]
        
        value_to_add = np.random.uniform(0.01,1.0)
        #draw a row randomly, which is not the last row
        idx = np.random.choice(list(range(0,len(lastcol)-1)),
                               size=1)[0]
        #Add the probability to the i'th row
        lastcol[idx] = value_to_add
        # write the changes back to matrix P
        P[:,len(D_orig)-1] = lastcol
        
        #normalize the row probabilities
        for row in range(0,len(lastcol)):
            P[row,:] = P[row,:]/np.sum(P[row,:])
        
        #get colsum again
        colsum = np.sum(P[:,len(D_orig)-1])

    #Test3: Make sure the absorption state can be reached from every state
    # Test2 only ensures that some rows lead to absorption. A set of states with transitions among
    # themselves only (a closed class) can still exist, and a trace which enters it would never end.
    unreachable = States_not_reaching_absorption(P)

    while len(unreachable) > 0:
        #draw one of the states which cannot reach the absorption state
        idx = np.random.choice(unreachable)

        #add a random probability of a transition to the absorption state, as in Test2
        P[idx, len(D_orig)-1] = np.random.uniform(0.01,1.0)

        #normalize the row probabilities
        P[idx,:] = P[idx,:]/np.sum(P[idx,:])

        #check again, as the other states of the closed class may still not reach absorption
        unreachable = States_not_reaching_absorption(P)


    #generate a dataframe with labels for the "to" states
    P_df = pd.DataFrame(np.reshape(P, (len(D_orig),len(D_orig))))
    P_df.columns = D_orig
    #print(P_df)
    return P_df


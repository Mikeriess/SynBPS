# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:35:33 2021

@author: Mike
"""


def Generate_transition_matrix_med_ent(D = ["a","b","c","d","e"],
                                       n_tranitions=3,
                                       limit_trials=1000,
                                       kappa = 2):
    """
    Parameters
    ----------
    D : TYPE, optional
        A list of states, such as D=["a","b","c","d","e"].
    
    e_steps : TYPE, optional
        Max expected steps to absorption
        
    n_tranitions : TYPE, optional
       Total number of transitions allowed from each state
    
    limit_trials : TYPE, optional
        Max number of trials for random search

    Returns
    -------
    P : TYPE
        DESCRIPTION.

    """
    import sys

    #error handling
    if len(D) < n_tranitions:
        print("n_transitions cannot be larger than the statespace")
        sys.exit()
        
    if n_tranitions < 3:
        print("n_transitions is less than 3, which can cause problems for large state-spaces")
        
    if n_tranitions < 2:
        print("n_transitions is less than 2, which will lead to an endless search")
        sys.exit()
    
    
    ####################################
    
    def Get_steps(P,D):
        import numpy as np
        
        #generate the transient matrix
        Q = P[:len(D)-1,:len(D)-1]
        
        #get the determinant of P
        P_det = np.linalg.det(P)
        print("det:",P_det)
        
        #if P is invertible, calculate number of steps to absorption
        #if 0.15 < P_det < 1:
        if P_det > 0.0001 and P_det < 1:
            print("pass:",str(P_det))
            
        #if np.isfinite(np.linalg.cond(Q)) and P_det > 0:
            #print(Q)
            I = np.identity(len(Q))
            M = np.linalg.inv((I-Q))
            
            M = np.dot(M, np.ones(len(Q)))
            
            steps = np.max(M)
            
        else:
            steps = 9999
        
        return steps

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
            x_i = np.random.uniform(0,1,1)
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
    
    #Test2: Make sure _someting_ leads to the absorption state
    
    # last column should sum to more than 1
    #rows, columns
    colsum = np.sum(P[:,len(D_orig)-1]) 
    
    while colsum < kappa:
        #get the last column
        lastcol = P[:,len(D_orig)-1]
        
        value_to_add = np.random.uniform(0.01,1.0,1)
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
    
    
    #generate a dataframe with labels for the "to" states
    P_df = pd.DataFrame(np.reshape(P, newshape=(len(D_orig),len(D_orig))))
    P_df.columns = D_orig
    #print(P_df)
    return P_df


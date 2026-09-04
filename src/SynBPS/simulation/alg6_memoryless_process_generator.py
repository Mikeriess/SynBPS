# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:22:42 2021

@author: Mike
"""

def Process_without_memory(D = ["a","b","c","d","e"], 
                        mode = ["min_entropy","max_entropy","med_entropy", "custom"][2], 
                        num_traces=2,
                        num_transitions=5, 
                        custom_distribution=None,
                        seed_value=1337):
    
    import numpy as np
    import pandas as pd
    
    #the memoryless process draws from the global numpy stream, which is seeded here for reproducibility
    np.random.seed(seed_value)
    
    from SynBPS.simulation.Memoryless_process.alg2_initial_probabilities import GenerateInitialProb

    from SynBPS.simulation.Memoryless_process.alg3_transition_matrix_min_entropy import Generate_transition_matrix_min_ent
    from SynBPS.simulation.Memoryless_process.alg4_transition_matrix_max_entropy import Generate_transition_matrix_max_ent
    from SynBPS.simulation.Memoryless_process.alg5_transition_matrix_med_entropy import Generate_transition_matrix_med_ent
  
    #error handling
    if mode not in ["min_entropy","max_entropy","med_entropy","custom"]:
        raise ValueError("mode must be min_entropy, max_entropy, med_entropy or custom. Change process_entropy in the settings.")
    
    #mode = ["min_entropy","max_entropy","med_entropy"][1]
    repetitions = num_traces

    #D = ["a","b","c","d","e"]
    D_abs = D.copy()
    D_abs.append("END")

    # Eventlog
    Theta = []
    
    if mode =="min_entropy":
        #initial probabilities
        P0 = GenerateInitialProb(D, p0_type="min_entropy")
        P = Generate_transition_matrix_min_ent(D, P0)
        P.index = D_abs
        
    if mode =="max_entropy":
        #initial probabilities
        P0 = GenerateInitialProb(D, p0_type="max_entropy")
        P = Generate_transition_matrix_max_ent(D)
        P.index = D_abs
        
    if mode =="med_entropy":
        #initial probabilities
        P0 = GenerateInitialProb(D, p0_type="med_entropy")
        P = Generate_transition_matrix_med_ent(D, n_tranitions=num_transitions)
        P.index = D_abs

    if mode =="custom":
        if mode == "custom" and custom_distribution is None:
            raise ValueError("process_entropy is custom, but no custom_distributions were given. Add custom_distributions with the files p0, p and Lambda to the settings.")
        #initial probabilities
        P0 = pd.read_csv(custom_distribution["p0"])["p0"].tolist()
        if len(P0) != len(D):
            raise ValueError("The p0 file must have one row per activity (statespace_size rows). Change the p0 file or statespace_size.")
        
        #transition matrix
        P = pd.read_csv(custom_distribution["p"])
        if len(P) != len(D_abs):
            raise ValueError("The p file must have one row per activity plus one for the absorption state (statespace_size + 1 rows). Change the p file or statespace_size.")
        if len(P.columns) != len(D_abs):
            raise ValueError("The p file must have one column per activity plus one for the absorption state (statespace_size + 1 columns). Change the p file or statespace_size.")
        P.columns = D_abs
        P.index = D_abs

    
    # Transition matrices
    Phi = [P0, P]
        
    for trace in list(range(0,repetitions)):
            
        #placeholder for trace
        sigma = []
        
        #counter
        t=1
        
        #sample from initial distribution
        e_t = np.random.choice(D, #len(D), #
                               size=1, replace=False, p=P0)[0]
        
        #append first event to trace
        sigma.append(e_t)
        
        while e_t != D_abs[len(D_abs)-1]:
            t = t+1
        
            #get conditional probability (e_t'th row of P)
            p_t = P.loc[P.index==e_t]
                        
            e_t = np.random.choice(D_abs, size=1, replace=False, p=p_t.values[0])[0]
            
            sigma.append(e_t)
        
        #print("trace:",sigma)

        Theta.append(sigma)
    return Theta, Phi
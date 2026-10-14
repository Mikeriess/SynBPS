# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:22:42 2021

@author: Mike
"""

def Process_without_memory(D = ["a","b","c","d","e"], 
                        mode = ["min_entropy","max_entropy","med_entropy", "custom"][2], 
                        num_traces=2,
                        num_transitions=5, 
                        kappa=2,
                        custom_distribution=None,
                        max_len=10000,
                        seed_value=1337):
    """
    Parameters
    ----------
    D : statespace (without absorption state)
    mode : min_entropy, med_entropy, max_entropy or custom
    num_traces : number of traces to generate
    num_transitions : number of possible transitions from each state (med_entropy only)
    kappa : minimum total probability of absorption over all states (med_entropy only), see alg 5
    custom_distribution : dict with the files p0, p and Lambda (custom only)
    max_len : maximum number of events in a trace before an exception is raised
    seed_value : seed of the two random streams of the process: memoryless_tables for P0 and P
        (alg. 2 to 5) and memoryless_sampling for the traces, so the tables do not depend on
        num_traces, and the traces of a seed do not depend on the global numpy stream

    Returns
    -------
    Theta : list of traces, each ending with the absorption state END
    HOMC : dict with the process in the structure of create_homc (the process with memory, of order K = 1):
        D, D_abs, P0 (array over D), Phi ({1: {(d,): probability vector over D_abs}}), K, mode,
        and P, the transition matrix as a dataframe with the rows and columns D_abs
    """

    import numpy as np
    import pandas as pd
    from SynBPS.simulation.simulation_helpers import make_rng
    
    #one stream for the tables and one for the sampling, as alg. 7 does for the process with memory
    rng_tables = make_rng(seed_value, "memoryless_tables")
    rng_sampling = make_rng(seed_value, "memoryless_sampling")
    
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
        P0 = GenerateInitialProb(D, p0_type="min_entropy", rng=rng_tables)
        P = Generate_transition_matrix_min_ent(D, P0, rng=rng_tables)
        P.index = D_abs
        
    if mode =="max_entropy":
        #initial probabilities
        P0 = GenerateInitialProb(D, p0_type="max_entropy", rng=rng_tables)
        P = Generate_transition_matrix_max_ent(D, rng=rng_tables)
        P.index = D_abs
        
    if mode =="med_entropy":
        #initial probabilities
        P0 = GenerateInitialProb(D, p0_type="med_entropy", rng=rng_tables)
        P = Generate_transition_matrix_med_ent(D, n_tranitions=num_transitions, kappa=kappa, rng=rng_tables)
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

    
    # the process in the structure of create_homc: Phi[1][(d,)] is the row of d, as for a HOMC of order 1
    P0 = np.asarray(P0, dtype=float)
    HOMC = {"D":D, 
            "D_abs":D_abs, 
            "P0":P0, 
            "Phi":{1: {(d,): P.loc[d].values.astype(float) for d in D}}, 
            "K":1, 
            "mode":mode, 
            "P":P}
        
    for trace in list(range(0,repetitions)):
            
        #placeholder for trace
        sigma = []
        
        #counter
        t=1
        
        #sample from initial distribution
        e_t = D[rng_sampling.choice(len(D), p=P0)]
        
        #append first event to trace
        sigma.append(e_t)
        
        while e_t != D_abs[len(D_abs)-1]:
            #Make sure the process cannot run forever (same guard as alg 7)
            if len(sigma) >= max_len:
                raise Exception("Trace did not reach the absorption state within max_len events. Increase max_len, or change the transition matrix if it is custom.")

            t = t+1
        
            #get conditional probability (e_t'th row of P)
            p_t = P.loc[P.index==e_t]
                        
            e_t = D_abs[rng_sampling.choice(len(D_abs), p=p_t.values[0])]
            
            sigma.append(e_t)
        
        #print("trace:",sigma)

        Theta.append(sigma)
    return Theta, HOMC

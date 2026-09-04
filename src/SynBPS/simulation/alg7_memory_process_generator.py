# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:10:31 2021

@author: Mike
"""

"""
Generation of transition tables and traces for the process with memory (HOMC of order K).
"""

def Generate_context_probabilities(D_abs = ["a","b","c","d","e","END"], 
                                   mode = ["min_entropy","max_entropy","med_entropy"][2], 
                                   n_transitions=3, 
                                   p_abs_min=0.05, 
                                   rng=None):
    """
    Conditional probabilities P(e_t | context) for one context (row) of the HOMC.
    Same logic as alg 4 and 5 for the memoryless process, but for one row at a time,
    as the number of rows grows with |D|^K.
    
    Parameters
    ----------
    D_abs : statespace incl. the absorption state (last element)
    mode : max_entropy or med_entropy (min_entropy rows are set in create_homc)
    n_transitions : number of possible transitions from the context (med_entropy only)
    p_abs_min : minimum probability of absorption from the context (0 = no guarantee)
    rng : numpy random generator

    Returns
    -------
    L : probability vector over D_abs
    """
    import numpy as np
    
    #initialize L as a zero vector of length D_abs
    L = np.zeros(len(D_abs))
    
    #index of the absorption state
    abs_idx = len(D_abs)-1
    
    if mode == "max_entropy":
        #all transitions are possible, with random weights (alg 4)
        L = rng.uniform(0,1,len(D_abs))
    
    if mode == "med_entropy":
        #draw n states from D_abs, without replacement (alg 5)
        selected = rng.choice(len(D_abs), size=n_transitions, replace=False)
        
        #draw probabilities for each selected state
        L[selected] = rng.uniform(0,1,n_transitions)
    
    #normalize to probability space
    L = L/np.sum(L)
    
    #Test1: Make sure _something_ leads to the absorption state from every context
    #(the HOMC equivalent of kappa in alg 5: P(trace length > l) <= (1-p_abs_min)^l)
    if p_abs_min > 0 and L[abs_idx] < p_abs_min:
        #scale the other states down, so that the absorption state gets p_abs_min
        L = L*(1-p_abs_min)/(np.sum(L)-L[abs_idx])
        L[abs_idx] = p_abs_min
        
        #normalize again
        L = L/np.sum(L)
    
    return L


"""
############################################################
"""


def create_homc(D = ["a","b","c","d","e"], 
                K=2, 
                mode = ["min_entropy","max_entropy","med_entropy"][2], 
                n_transitions=3, 
                p_abs_min=0.05, 
                seed_value=1337):
    """
    Generate the initial probabilities P0 and the transition tables Phi = {P1,...,PK}
    of increasing order, as in eq. (4) and alg 7 in the paper.
    Phi[i][context] is the probability vector over D_abs, given the last i states.
    
    Parameters
    ----------
    D : statespace (without absorption state)
    K : order of the HOMC (number of previous states to condition on)
    mode : min_entropy, med_entropy or max_entropy
    n_transitions : number of possible transitions from each context (med_entropy only)
    p_abs_min : minimum probability of absorption from any context (med_entropy and max_entropy only)
    seed_value : seed for the random generator

    Returns
    -------
    HOMC : dict with D, D_abs, P0, Phi, K and mode
    """
    import numpy as np
    import itertools
    
    #error handling
    if K < 1:
        raise ValueError("process_memory (K) must be 1 or larger for the process with memory. Use process_type memoryless for a process without memory.")
    
    if mode not in ["min_entropy","med_entropy","max_entropy"]:
        raise ValueError("process_entropy must be min_entropy, med_entropy or max_entropy for the process with memory. Custom distributions are only supported for process_type memoryless.")
    
    if mode == "med_entropy" and (n_transitions < 2 or n_transitions > len(D)+1):
        raise ValueError("med_ent_n_transitions must be between 2 and the statespace size plus 1 (the absorption state). Change med_ent_n_transitions or statespace_size.")
    
    if p_abs_min < 0 or p_abs_min >= 1:
        raise ValueError("p_abs_min must be between 0 and 1 (1 excluded). Set p_abs_min to 0 to disable the absorption guarantee.")
    
    #one random generator for all tables (reproducible from seed_value)
    rng = np.random.default_rng(seed_value)
    
    # Including absorption state
    D_abs = D.copy()
    D_abs.append("END")
    
    ##### Part 1: Initial probabilities
    
    #P0 is drawn over D only, as a trace can never start in the absorption state
    if mode == "min_entropy":
        # P0 is one-hot (alg 2)
        P0 = np.zeros(len(D))
        P0[rng.integers(len(D))] = 1
        
    if mode != "min_entropy":
        #draw from uniform dist and normalize (alg 2)
        P0 = rng.uniform(0,1,len(D))
        P0 = P0/np.sum(P0)
    
    ##### Part 2: Transition tables of increasing order
    
    #transition tables Phi = {P1,...,PK}
    Phi = {}
    
    if mode == "min_entropy":
        """
        Minimum entropy: one deterministic trace (same idea as alg 3)
            With memory, activities may repeat, but a context of K states can only
            be visited once, otherwise the deterministic process would never end.
        """
        
        #trace length (as in alg 3, where the trace is a permutation of D)
        trace_len = len(D)
        
        #start in the initial state
        trace = [D[int(np.argmax(P0))]]
        
        #contexts already visited
        seen = set()
        
        while len(trace) < trace_len:
            #the last K states (fewer in the beginning of the trace)
            context = tuple(trace[-K:])
            
            #mark the context as visited
            seen.add(context)
            
            #candidate next states, which do not lead to an already visited context
            candidates = [d for d in D if tuple((trace+[d])[-K:]) not in seen]
            
            #stop if no candidates are left
            if len(candidates) == 0:
                break
            
            #draw the next state
            trace.append(candidates[rng.integers(len(candidates))])
        
        #end the trace with the absorption state
        trace.append("END")
        
        #contexts which are not on the trace can never be reached: they lead to the absorption state
        for i in range(1,K+1):
            Phi[i] = {}
            for context in itertools.product(D, repeat=i):
                R = np.zeros(len(D_abs))
                R[len(D_abs)-1] = 1
                Phi[i][context] = R
        
        #overwrite the rows along the trace with the deterministic transitions
        for t in range(1,len(trace)):
            #order i is smaller than K in the beginning of the trace
            i = min(t,K)
            context = tuple(trace[t-i:t])
            
            #generate vector
            R = np.zeros(len(D_abs))
            R[D_abs.index(trace[t])] = 1
            Phi[i][context] = R
    
    if mode != "min_entropy":
        #for each order 1..K, generate a table over all contexts of length i
        for i in range(1,K+1):
            Phi[i] = {}
            
            #contexts do not include the absorption state, as the trace stops there
            for context in itertools.product(D, repeat=i):
                Phi[i][context] = Generate_context_probabilities(D_abs, 
                                                                 mode=mode, 
                                                                 n_transitions=n_transitions, 
                                                                 p_abs_min=p_abs_min, 
                                                                 rng=rng)
    
    HOMC = {"D":D, "D_abs":D_abs, "P0":P0, "Phi":Phi, "K":K, "mode":mode}
    
    return HOMC


"""
############################################################
"""


def Process_with_memory(D = ["a","b","c","d","e"], 
                        mode = ["min_entropy","max_entropy","med_entropy"][2], 
                        num_traces=2, 
                        K=2,
                        num_transitions=3, 
                        p_abs_min=0.05,
                        max_len=10000,
                        seed_value=1337):
    """
    Parameters
    ----------
    D : statespace (without absorption state)
    mode : min_entropy, med_entropy or max_entropy
    num_traces : number of traces to generate
    K : order of the HOMC
    num_transitions : number of possible transitions from each context (med_entropy only)
    p_abs_min : minimum probability of absorption from any context
    max_len : maximum number of events in a trace before an exception is raised
    seed_value : seed for the random generator

    Returns
    -------
    Theta : list of traces, each ending with the absorption state END
    HOMC : dict with the initial probabilities P0 and the transition tables Phi
    """
    import numpy as np
    
    ##### Part 1: Generate the transition probabilities
    
    # event-log container
    Theta = []
    
    #create the markov chain
    HOMC = create_homc(D, 
                       K=K, 
                       mode=mode, 
                       n_transitions=num_transitions, 
                       p_abs_min=p_abs_min, 
                       seed_value=seed_value)
    
    D_abs = HOMC["D_abs"]
    P0 = HOMC["P0"]
    Phi = HOMC["Phi"]
    
    #random generator for the sampling (separate stream, but from the same seed)
    rng = np.random.default_rng(seed_value+1)
    
    ##### Part 2: Draw from the distributions
    
    for trace in list(range(0,num_traces)):
        
        #Trace placeholder
        Q = []
        
        #sample from initial distribution
        e_t = D[rng.choice(len(D), p=P0)]
        
        #append first event to trace
        Q.append(e_t)
        
        #Continue drawing until the absorption state is reached
        while e_t != "END":
            
            #use order t until K previous states are available, then order K (alg 7)
            i = min(len(Q),K)
            
            #get conditional probability of the last i states
            context = tuple(Q[-i:])
            p_t = Phi[i][context]
            
            #draw the next state
            e_t = D_abs[rng.choice(len(D_abs), p=p_t)]
            
            Q.append(e_t)
            
            #Test2: Make sure the process cannot run forever
            if len(Q) >= max_len:
                raise Exception("Trace did not reach the absorption state within max_len events. Increase p_abs_min or max_len.")
        
        #Update the event-log
        Theta.append(Q)
    
    #print("generated traces:", len(Theta))
    return Theta, HOMC

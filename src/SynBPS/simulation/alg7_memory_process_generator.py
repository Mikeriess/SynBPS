# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:10:31 2021

@author: Mike
"""

def Process_with_memory(D = ["a","b","c","d","e"], 
                                    mode = ["min_entropy","max_entropy","med_entropy"][2], 
                                    num_traces=2, 
                                    sample_len=100,
                                    K=2,
                                    num_transitions=5):
    import numpy as np
    import pandas as pd
    import sys
    
    ##### Part 1: Generate the transition probabilities
    
    # event-log container
    Theta = []
    
    # Including absorption state
    D_abs = D.copy()
    D_abs.append("!")
    
    
    # Generate the model
    from SynBPS.simulation.alg2_initial_probabilities import GenerateInitialProb
    from SynBPS.simulation.homc_helpers import create_homc

    
    #generate initial probabilities
    probabilities = GenerateInitialProb(D_abs, p0_type=mode)    
    P0 = {}
    
    for i in range(0,len(D_abs)):
        P0.update({D_abs[i]:probabilities[i]})
    

    #print("mode",mode)
    #create the markov chain
    HOMC = create_homc(D_abs, P0, h=K, mode=mode, n_transitions=num_transitions)
    
    ##### Part 2: Draw from the distributions
    while len(Theta) != num_traces:
        #for trace in list(range(0,num_traces)):
                
        #Trace placeholder
        Q = []
        
        trials = 0
        
        #Continue drawing until there is an absorption event when length = x
        while "!" not in set(Q): #or len(Q) > 1
            #counter
            trials = trials + 1

            if trials > 1:
                #print("trial",trials)
                #Sample trace from model
                new_samplelen = int(sample_len)*(trials*10)
                Q = HOMC.sample(new_samplelen)
            else:
                #Sample trace from model
                Q = HOMC.sample(sample_len)

            
            #if absorption state is observed, remove all extra occurrences of it
            if "!" in set(Q):
                Q = Q[:Q.index('!')+1]

            #if only the absorbing state is observed, try again
            #if len(Q) == 1:
            #    print("trace:",trace,"only the absorbing state is observed, trying again")
            #    Q = [] 
            
            if trials > 10:
                Q = []
                print("Sequence did not reach absorbing state after 10 trials. Trying again.")
                break
            
        #recode the name of the termination event
        Q = [w.replace('!', 'END') for w in Q]
        
        #if there is more than one event
        if len(Q) > 1:
            #Update the event-log
            Theta.append(Q)

    print("generated traces:", len(Theta))
    return Theta, HOMC


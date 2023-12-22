# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:09:09 2022

@author: Mike
"""


def cartesian_product(a,b):
    import itertools

    c = list(itertools.product(a, b))
    return c


def combine_to_list(c):
    from SynBPS.simulation.simulation_helpers import flatten
    
    # combine the letters into one item
    newlist = []
    
    for i in range(0,len(c)):
        combination = flatten(c[i])
        newlist.append(combination)
            
    return newlist

def modify_to_absorption(c):
    """
    iterate over each line, and if E occurs at any point
    """
    newlist = []
    
    return newlist


def modify_rules(parent, states):
    import numpy as np
    #append probabilities to each row in the condition table
    condprob=[]
        
    #for each parent state
    for parentstate in states:
        
        #subset all rows starting with parent state i
        subset = [row for row in parent if row[0] == parentstate]

        """# manipulate the list """
        
        #All rows, starting with E, should lead only to E
        #If a sequence has E at any point, every subsequent entry becomes E
        
        new_subset = []
        
        for row in subset:
            
            #make a new row, based on rules
            newrow=[]
            
            #flag-variable
            e_observed = False
            
            #for each step in the sequence
            for idx in range(0,len(row)):
                
                
                # if e is observed in current timestep, set flag to true
                if row[idx] == "E":
                    e_observed = True
                
                # 
                if e_observed == True:
                    value = "E"
                else:
                    value = row[idx]
                
                #append new value, based on above logic
                newrow.append(value)
                
                                
            #append new modified row
            new_subset.append(newrow)
        
        #append to final list
        condprob = condprob + new_subset
    
    return condprob


def generate_condprob(parent, states, mode="max_entropy", n_transitions=5):
    import numpy as np
    #append probabilities to each row in the condition table
    condprob=[]
        
    #for each parent state
    for parentstate in states:
        
        #subset all rows starting with parent state i
        subset = [row for row in parent if row[0] == parentstate]

        """# manipulate the list """
        
        #All rows, starting with E, should lead only to E
        #If a sequence has E at any point, every subsequent entry becomes E
        
        if mode=="max_entropy":
            #get list of probabilities for each state
            vec = np.random.random(len(subset))
        
        if mode=="med_entropy":
            #get n random rows with probability > 0, and 0 for rest of the rows
            vec = np.zeros(len(subset)).tolist()
            
            ids = list(range(0,len(vec)))
            import random
            selected = random.sample(ids, n_transitions)
            
            for i in selected:
                vec[i] = np.round(np.random.random(1)[0],decimals=8)
                
        if mode=="min_entropy":
            #get 1 random row with probability == 1 and 0 for rest of the rows
            vec = np.zeros(len(subset)).tolist()
            
            ids = list(range(0,len(vec)))
            import random
            selected = random.sample(ids, 1)[0]
            
            #set probability to 1
            vec[selected] = 1
            
        
        #normalize it
        vec = np.round(vec/np.sum(vec), decimals=5)
        vec = vec.tolist()
        
        for i in range(0,len(subset)):
            #get the probability
            p = vec[i]
            
            #append it to row i in subset
            subset[i].append(p)
            
        #"""
        #append to final list
        condprob = condprob + subset
    
    return condprob

def create_homc(states, h0, h=2, mode="max_entropy", n_transitions=5):
        
    from SynBPS.simulation.homc_helpers import cartesian_product, combine_to_list, modify_rules, generate_condprob
    
    
    ######################################
    # P1
    
    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)
    
    #final steps
    g = modify_rules(d, states)
    p1_input = generate_condprob(g, states, mode, n_transitions)
    
    ######################################
    # P2
    
    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)
    
    e = cartesian_product(c, states)
    f = combine_to_list(e)
    
    #final steps
    g = modify_rules(f, states)
    p2_input = generate_condprob(g, states, mode, n_transitions)
    
    ######################################    
    # P3
    
    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)
    
    e = cartesian_product(c, d)
    f = combine_to_list(e)
    
    #final steps
    g = modify_rules(f, states)
    p3_input = generate_condprob(g, states, mode, n_transitions)
    
    ######################################    
    # P4
    
    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)
    
    e = cartesian_product(c, d)
    f = combine_to_list(e)
    
    e = cartesian_product(f, states)
    f = combine_to_list(e)
    
    #final steps
    g = modify_rules(f, states)
    p4_input = generate_condprob(g, states, mode, n_transitions)

    ######################################    
    # P5
    
    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)
    
    e = cartesian_product(c, d)
    f = combine_to_list(e)
    
    e = cartesian_product(f, states)
    f = combine_to_list(e)
    
    #final steps
    g = modify_rules(f, states)
    p4_input = generate_condprob(g, states, mode, n_transitions)

    """
    Input generated tables to pomegranate
    """
    from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, MarkovChain
    
    if h == 1:
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        HOMC = MarkovChain([p0, p1])
        
    if h == 2:
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        p2 = ConditionalProbabilityTable(p2_input, [p1])
        
        HOMC = MarkovChain([p0, p1, p2])
        
    if h == 3:
        
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        p2 = ConditionalProbabilityTable(p2_input, [p1])
        
        p3 = ConditionalProbabilityTable(p3_input, [p2])
        
        HOMC = MarkovChain([p0, p1, p2, p3])
        
    if h == 4:
         
        p0 = DiscreteDistribution(h0)
         
        p1 = ConditionalProbabilityTable(p1_input, [p0])
         
        p2 = ConditionalProbabilityTable(p2_input, [p1])
         
        p3 = ConditionalProbabilityTable(p3_input, [p2])
         
        p4 = ConditionalProbabilityTable(p4_input, [p3])
         
        HOMC = MarkovChain([p0, p1, p2, p3, p4])
         
    if h > 4:
        print("h > 4 not supported yet - please create an issue on github")
        HOMC = 0
    
    return HOMC
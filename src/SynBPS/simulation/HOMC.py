import numpy as np

def GenerateHOMC(D = ["a","b","c","d"], # statespace for intial prob
                 mode = ["min_entropy","med_entropy","max_entropy"][0], # complexity
                n_transitions = 2 # number of transitions with med_entropy
                 ):
    """
    Create a higher-order markov chain
    """
    #synbps.simulation.
    from homc_helpers import cartesian_product, combine_to_list, modify_rules, generate_condprob
    from alg2_initial_probabilities import GenerateInitialProb

    # A function to list-based conditional probabilities into a nested dictionary:
    def transform_markov_chain(markov_chain_list):
        markov_chain_dict = {}
        for table in markov_chain_list:
            order = len(table[0]) - 2
            transitions = {}
            for row in table:
                state, next_state, probability = tuple(row[:-2]), row[-2], row[-1]
                if state not in transitions:
                    transitions[state] = {}
                transitions[state][next_state] = probability
            markov_chain_dict[order] = transitions
        return markov_chain_dict
    
    # statespace for conditional prob
    D_abs = D.copy()
    D_abs.append("!")

    # Unconditional probabilities, excluding the absorbing state
    P0 = GenerateInitialProb(D=D, 
                            p0_type="regular")

    # copy for use in higher order probability tables
    states = D_abs.copy()

    ######################################
    # P1

    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)

    #final steps
    g = modify_rules(d, states)
    P1 = generate_condprob(g, states, mode, n_transitions)

    ######################################
    # P2

    #for each link
    c = cartesian_product(states, states)
    d = combine_to_list(c)

    e = cartesian_product(c, states)
    f = combine_to_list(e)

    #final steps
    g = modify_rules(f, states)
    P2 = generate_condprob(g, states, mode, n_transitions)

    ######################################

    # final probability tables
    HOMC = [P1, P2]

    # convert to dictionary
    P_k = transform_markov_chain(HOMC)

    return P0, P_k


def SampleHOMC(D, P0, P_k):
    """
    Sample from the higher-order markov chain
    """

    # placeholder for trace
    sigma = []

    # stop when absorbing state is reached
    while "!" not in set(sigma):
        
        # determine tracelength
        tracelen = len(sigma)
        
        if tracelen == 0:
            #sample first event from P0
            e_t = np.random.choice(D,
                            size=1, 
                            replace=False, 
                            p=P0)[0]
            # add first event to trace
            sigma.append(e_t)
        
        # if length is less than or equal K-1, use K'th order probability table
        if tracelen > 0 and tracelen <= max(P_k.keys()):
            # retrieve the order to subset P_k from
            order = len(sigma)
        
            # retrieve the probability distribution
            prob_dist = P_k[order][tuple(sigma)]
        
            # Extract elements and their associated probabilities
            elements = list(prob_dist.keys())
            probabilities = list(prob_dist.values())
            
            # Use np.random.choice with the probabilities
            e_t = np.random.choice(elements,
                                size=1, 
                                replace=False, 
                                p=probabilities)
            
            # add sampled event to trace
            sigma.append(e_t[0])
        
        # if length is > K-1, reach the absorbing state
        if tracelen > max(P_k.keys()):
            # truncate/end everything after K
            e_t = "!"
            
            # add sampled event to trace
            sigma.append(e_t)

    return sigma

############ Testing
D = ["a","b","c","d"]

P0, P_k = GenerateHOMC(D, # statespace for intial prob
                    mode = ["min_entropy","med_entropy","max_entropy"][1], # complexity
                    n_transitions = 2 # number of transitions with med_entropy
                    )

sigma = SampleHOMC(D, P0, P_k)
print(sigma)
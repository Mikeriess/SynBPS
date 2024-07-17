import numpy
import json
import itertools as it

from SynBPS.simulation.Memory_process.distributions.Distribution import Distribution
from SynBPS.simulation.Memory_process.distributions.DiscreteDistribution import DiscreteDistribution
from SynBPS.simulation.Memory_process.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable

class MarkovChain(object):
    """A Markov Chain.

    Implemented as a series of conditional distributions, the Markov chain
    models P(X_i | X_i-1...X_i-k) for a k-th order Markov network. The
    conditional dependencies are directly on the emissions, and not on a
    hidden state as in a hidden Markov model.

    Parameters
    ----------
    distributions : list, shape (k+1)
        A list of the conditional distributions which make up the markov chain.
        Begins with P(X_i), then P(X_i | X_i-1). For a k-th order markov chain
        you must put in k+1 distributions.

    Attributes
    ----------
    distributions : list, shape (k+1)
        The distributions which make up the chain.

    Examples
    --------
    >>> d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
    >>> d2 = ConditionalProbabilityTable([['A', 'A', 0.33],
                                          ['B', 'A', 0.67],
                                          ['A', 'B', 0.82],
                                          ['B', 'B', 0.18]], [d1])
    >>> mc = MarkovChain([d1, d2])
    """

    def __init__(self, distributions,random_state=None):
        self.k = len(distributions) - 1
        self.distributions = distributions
        self.random_state = numpy.random.RandomState(random_state)

    def __reduce__(self):
        return self.__class__, (self.distributions,)


    def sample(self, length):
        """Create a random sample from the model.

        Parameters
        ----------
        length : int or Distribution
            Give either the length of the sample you want to generate, or a
            distribution object which will be randomly sampled for the length.
            Continuous distributions will have their sample rounded to the
            nearest integer, minimum 1.

        Returns
        -------
        sequence : array-like, shape = (length,)
            A sequence randomly generated from the markov chain.
        """
        if isinstance(length, Distribution):
            length = int(length.sample())
        length = max(length, 1)

        sequence = [self.distributions[0].sample(random_state=self.random_state)]
        if length == 1:
            return sequence

        for j, distribution in enumerate(self.distributions[1:]):
            parents = {self.distributions[l]: sequence[l] for l in range(j+1)}
            sequence.append(distribution.sample(parents, random_state=self.random_state))

        if len(sequence) == length:
            return sequence

        for l in range(length - len(sequence)):
            parents = {self.distributions[k]: sequence[l+k+1] for k in range(self.k)}
            sequence.append(self.distributions[-1].sample(parents, random_state=self.random_state))

        return sequence

    def fit(self, sequences, weights=None, inertia=0.0):
        """Fit the model to new data using MLE.

        The underlying distributions are fed in their appropriate points and
        weights and are updated.

        Parameters
        ----------
        sequences : array-like, shape (n_samples, variable)
            This is the data to train on. Each row is a sample which contains
            a sequence of variable length
        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample. If nothing is passed in then
            each sample is assumed to be the same weight. Default is None.
        inertia : float, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be
            old_param*inertia + new_param*(1-inertia), so an inertia of 0 means
            ignore the old parameters, whereas an inertia of 1 means ignore the
            new parameters. Default is 0.0.

        Returns
        -------
        None
        """
        self.summarize(sequences, weights)
        self.from_summaries(inertia)

    def summarize(self, sequences, weights=None):
        """Summarize a batch of data and store sufficient statistics.

        This will summarize the sequences into sufficient statistics stored in
        each distribution.

        Parameters
        ----------
        sequences : array-like, shape (n_samples, variable)
            This is the data to train on. Each row is a sample which contains
            a sequence of variable length
        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample. If nothing is passed in then
            each sample is assumed to be the same weight. Default is None.

        Returns
        -------
        None
        """
        if weights is None:
            weights = numpy.ones(len(sequences), dtype='float64')
        else:
            weights = numpy.asarray(weights)

        n = max(map(len, sequences))

        for i in range(self.k):
            if i == 0:
                symbols = [sequence[0] for sequence in sequences]
            else:
                symbols = [sequence[:i+1]
                           for sequence in sequences if len(sequence) > i]
            self.distributions[i].summarize(symbols, weights)

        for j in range(n-self.k):
            if self.k == 0:
                symbols = [sequence[j]
                           for sequence in sequences
                           if len(sequence) > j+self.k]
            else:
                symbols = [sequence[j:j+self.k+1]
                           for sequence in sequences
                           if len(sequence) > j+self.k]
            self.distributions[-1].summarize(symbols, weights)

    def from_summaries(self, inertia=0.0):
        """Fit the model to the collected sufficient statistics.

        Fit the parameters of the model to the sufficient statistics gathered
        during the summarize calls. This should return an exact update.

        Parameters
        ----------
        inertia : float, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be
            old_param*inertia + new_param * (1-inertia), so an inertia of 0
            means ignore the old parameters, whereas an inertia of 1 means
            ignore the new parameters. Default is 0.0.

        Returns
        -------
        None
        """
        for i in range(self.k+1):
            self.distributions[i].from_summaries(inertia=inertia)

    def to_json(self, separators=(',', ' : '), indent=4):
        """Serialize the model to a JSON.

        Parameters
        ----------
        separators : tuple, optional
            The two separators to pass to the json.dumps function for
            formatting. Default is (',', ' : ').
        indent : int, optional
            The indentation to use at each level. Passed to json.dumps for
            formatting. Default is 4.

        Returns
        -------
        json : str
            A properly formatted JSON object.
        """
        model = {
            'class': 'MarkovChain',
            'distributions': [d.to_dict() for d in self.distributions]
        }

        return json.dumps(model, separators=separators, indent=indent)

    @classmethod
    def from_json(cls, s):
        """Read in a serialized model and return the appropriate classifier.

        Parameters
        ----------
        s : str
            A JSON formatted string containing the file.

        Returns
        -------
        model : object
            A properly initialized and baked model.
        """
        d = json.loads(s)
        distributions = [Distribution.from_dict(j)
                         for j in d['distributions']]
        model = cls(distributions)
        return model

    @classmethod
    def from_samples(cls, X, weights=None, k=1):
        """Learn the Markov chain from data.

        Takes in the memory of the chain (k) and learns the initial
        distribution and probability tables associated with the proper
        parameters.

        Parameters
        ----------
        X : array-like, list or numpy.array
            The data to fit the structure too as a list of sequences of
            variable length. Since the data will be of variable length,
            there is no set form
        weights : array-like, shape (n_nodes), optional
            The weight of each sample as a positive double. Default is None.
        k : int, optional
            The number of samples back to condition on in the model. Default
            is 1.

        Returns
        -------
        model : MarkovChain
            The learned markov chain model.
        """
        symbols = set()
        for seq in X:
            for symbol in seq:
                symbols.add(symbol)

        n = len(symbols)
        d = DiscreteDistribution({symbol: 1./n for symbol in symbols})
        distributions = [d]

        for i in range(1, k+1):
            table = []
            for key in it.product(symbols, repeat=i+1):
                table.append(list(key) + [1./n])
            d = ConditionalProbabilityTable(table, distributions[:])
            distributions.append(d)

        model = cls(distributions)
        model.fit(X)
        return model

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


def generate_condprob(parent, states, mode="max_entropy", n_transitions=5, seed_value=1337):
    import numpy as np
    #np.random.seed(seed_value)

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
            #import random
            #selected = random.sample(ids, n_transitions)
            selected = np.random.choice(ids, n_transitions, replace=False)
            
            for i in selected:
                vec[i] = np.round(np.random.random(1)[0],decimals=8)




            ######
            #selected = np.random.choice(len(subset), n_transitions, replace=False)
            #vec[selected] = np.random.random(n_transitions)
            #np.round(vec, decimals=8, out=vec)
                
        if mode=="min_entropy":
            #get 1 random row with probability == 1 and 0 for rest of the rows
            vec = np.zeros(len(subset)).tolist()
            
            ids = list(range(0,len(vec)))

            #import random
            #selected = random.sample(ids, 1)[0]
            selected = np.random.choice(ids, 1, replace=False)[0] # this last index might be removed

            # set probability to 1
            vec[selected] = 1
            
            #####
            #vec = np.zeros(len(subset))
            #selected = np.random.choice(n)
            #vec[selected] = 1
            #vec = vec.tolist()
        
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

def create_homc(states, h0, h=2, mode="max_entropy", n_transitions=5, seed_value=1337):
    import numpy as np
    #np.random.seed(seed_value)
        
    from SynBPS.simulation.Memory_process.MarkovChain import cartesian_product, combine_to_list, modify_rules, generate_condprob

    from SynBPS.simulation.Memory_process.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
    from SynBPS.simulation.Memory_process.distributions.DiscreteDistribution import DiscreteDistribution
    from SynBPS.simulation.Memory_process.MarkovChain import MarkovChain
    
    
    ######################################
    # P1
    
    #for each link
    #c = cartesian_product(states, states)
    #d = combine_to_list(c)
    
    #final steps
    #g = modify_rules(d, states)
    #p1_input = generate_condprob(g, states, mode, n_transitions, seed_value)
    
    ######################################
    # P2
    
    #for each link
    #c = cartesian_product(states, states)
    #d = combine_to_list(c)
    
    #e = cartesian_product(c, states)
    #f = combine_to_list(e)
    
    #final steps
    #g = modify_rules(f, states)
    #p2_input = generate_condprob(g, states, mode, n_transitions, seed_value)
    
    ######################################    
    # P3
    
    #for each link
    #c = cartesian_product(states, states)
    #d = combine_to_list(c)
    
    #e = cartesian_product(c, d)
    #f = combine_to_list(e)
    
    #final steps
    #g = modify_rules(f, states)
    #p3_input = generate_condprob(g, states, mode, n_transitions, seed_value)
    
    ######################################    
    # P4
    
    #for each link
    #c = cartesian_product(states, states)
    #d = combine_to_list(c)
    
    #e = cartesian_product(c, d)
    #f = combine_to_list(e)
    
    #e = cartesian_product(f, states)
    #f = combine_to_list(e)
    
    #final steps
    #g = modify_rules(f, states)
    #p4_input = generate_condprob(g, states, mode, n_transitions, seed_value)

    ######################################    
    # P5
    

    """
    Input generated tables to MarkovChain class
    
    
    if h == 1:
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        HOMC = MarkovChain([p0, p1], random_state=seed_value)
        
    if h == 2:
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        p2 = ConditionalProbabilityTable(p2_input, [p1])
        
        HOMC = MarkovChain([p0, p1, p2], random_state=seed_value)
        
    if h == 3:
        
        p0 = DiscreteDistribution(h0)
        
        p1 = ConditionalProbabilityTable(p1_input, [p0])
        
        p2 = ConditionalProbabilityTable(p2_input, [p1])
        
        p3 = ConditionalProbabilityTable(p3_input, [p2])
        
        HOMC = MarkovChain([p0, p1, p2, p3], random_state=seed_value)
        
    if h == 4:
         
        p0 = DiscreteDistribution(h0)
         
        p1 = ConditionalProbabilityTable(p1_input, [p0])
         
        p2 = ConditionalProbabilityTable(p2_input, [p1])
         
        p3 = ConditionalProbabilityTable(p3_input, [p2])
         
        p4 = ConditionalProbabilityTable(p4_input, [p3])
         
        HOMC = MarkovChain([p0, p1, p2, p3, p4], random_state=seed_value)
         
    if h > 4:
        print("h > 4 not supported yet - please create an issue on github")
        HOMC = 0
    """

    # Generate conditional probability tables for h-order markov chains
    def recursive_state_process(states, mode, n_transitions, iterations, seed_value=1337):
        def process_iteration(iter_num):
            if iter_num == 1:
                # Base case: P1 process
                c = cartesian_product(states, states)
                d = combine_to_list(c)
                g = modify_rules(d, states)
                return generate_condprob(g, states, mode, n_transitions, seed_value)
            else:
                # Recursive case: build on previous iteration
                prev_result = process_iteration(iter_num - 1)
                prev_states = [item[:-1] for item in prev_result]  # Remove probabilities
                c = cartesian_product(prev_states, states)
                d = combine_to_list(c)
                g = modify_rules(d, states)
                return generate_condprob(g, states, mode, n_transitions, seed_value)
        return process_iteration(iterations)


    # Start by converting the initial probabilities
    p0 = DiscreteDistribution(h0)

    # Initiate the list of distributions
    distributions = [p0]

    # Run the recursive generation of distribution tables to append
    for order in range(0, h-1):
        if order == 0:
            p_i_input = recursive_state_process(states, mode, n_transitions, iterations=h, seed_value=seed_value)

            p_i = ConditionalProbabilityTable(p_i_input, [p0])
            distributions.append(p_i)
        else:
            p_i_input = recursive_state_process(states, mode, n_transitions, iterations=h, seed_value=seed_value)

            p_i = ConditionalProbabilityTable(p_i_input, distributions[-1])

            distributions.append(p_i)


    HOMC = MarkovChain(distributions, random_state=seed_value)

    return HOMC
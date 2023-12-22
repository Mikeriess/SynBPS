# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:06:34 2021

@author: Mike
"""

"""
Generation of trace arrival times in a synthetic event-log.
"""

def Generate_trace_arrivals(lambd = 1, n_arrivals=10):
    import numpy as np
    #initialize
    theta = []
    z = []
    t = 0
    id_counter = 1
    
    while len(z) < n_arrivals:
        #draw arrival time
        x_i = np.random.exponential(lambd,1)[0]
        
        #get a caseid for the case
        case_id = id_counter
        #increase ID counter
        id_counter = id_counter +1
        
        #increase time variable t
        t = t+x_i
        
        # append the case to the eventlog
        theta.append({"id":case_id,
         "x_i":x_i,
         "starttime":t})
        
        # append arrivaltime to vector z
        z.append(t)
        
    return theta, z

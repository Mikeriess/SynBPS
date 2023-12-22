# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:46:30 2021

@author: Mike
"""
import numpy as np
import pandas as pd

def Generate_lambdas(D,t, lambd_range):
    """
    Generate lambda values for capital lambda matrix
    
    Matrix is:
        D x T
        
        D = statespace
        T = timesteps
    """
    
    Lambd = np.random.uniform(low=0.0001, high=lambd_range,size=(len(D)*t))
    Lambd = Lambd.reshape(len(D),t)
    
    Lambd = pd.DataFrame(Lambd).T
    Lambd.columns = D
    
    return Lambd

"""
############################################################
"""


def Resource_offset(h = 0, m = 0.15, p = 0.5, n = 3):
    """
    

    Parameters
    ----------
    h : beginning offset
    m : time between requests
    p : probability of getting an idle agent
    n : number of agents

    Returns
    -------
    h : TYPE
        DESCRIPTION.

    """
    
    
    
    """
    #initial value
    k = 0
    while k < 1:
    """  
    
    #Get number of trials before success
    k = np.random.binomial(n, p, size=1)[0]
    
    # Add penalty for every trial until success (of getting an agent)
    h = m*k
    
    """
    #add time penalty of no agent available
    if k < 1:
        h=h+m
    """  
    return h


def TimeSinceMonday(z_i, t, y, m, u):
    """

    Parameters
    ----------
    z_i : arrival time of the i'th trace

    t : current timestep

    y : vector of all preceeding durations in the trace
    
    m : Resource-related offset.
    
    u : the number of time-units within a single week
        day: 7
        hours: 7*24 = 168
        minutes: 7*24*60 = 10 080

    Returns
    -------
    qt : the scheduled time to begin the case, since the beginning of the week.
    
    ##########################################
    test values
    
    z_i = 0
    t = 3
    #y = 0 #first case, leading 0
    y = [1,2,3] #first case, leading 0
    m = 0
    u = 168 # hours

    """
    
    
    if t == 0:
        """
        first event: y is a scalar 0
        """
        q_t = m + z_i #+ 1*(t>1)*(y)
        
    if t > 0:
        """
        not first event: y is a vector
        """
        
        #get all previous durations
        y_prev = []
        for j in list(range(0,t)): y_prev.append(y[j])
        y_prev_sum = np.sum(y_prev)
        
        
        #get the scheduled beginning:
        q_t = 1*(t>1)*(y_prev_sum)+m
    
    # get time since monday
    q_t = np.mod(q_t,u)
    
    return q_t



def Deterministic_offset(W, q_t):
    """

    Parameters
    ----------
    W : Rule-matrix with intervals that will result in a time-penalty
    q_t : Scheduled beinning time

    Returns
    -------
    b : Deterministic offset to the activity

    """
            
    W = np.array(W)
    W = np.reshape(W,(2, len(W[0]) )).T

    """
    # Testing the function
    i = 3
    print("val:",q_t,", from:",W[i,0],"to:",W[i,1])
    b = (W[i,0] <= q_t < W[i,1])*1*(W[i,1]-q_t)
    
    print("waiting time:",b)
    
    
    b = (W[:,0] <= q_t < W[:,1])*1
    """
    
    """
    Vectorized version
    """
            
    #evaluate logic and generate a binary vector 
    mask = ((W[:,0] <= q_t) & (q_t< W[:,1]))*1
    
    #calculate the time to get out of the closed period
    b = mask*(W[:,1]-q_t)
    
    #sum all the delays
    b = np.sum(b)
    
    return b
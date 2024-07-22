# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 22:16:41 2021

@author: Mike
"""

def Generate_time_variables(Theta = [["a","b","END"],                   #the generated eventlog
                                     ["a","b","END"],
                                     ["a","b","END"]],
                            
                                   D = ["a","b","c","d","e"],           #state space of the process
                                   
                            settings={"inter_arrival_time":1,           #lambda parameter of inter-arrival times
                                         
                                      "process_stability_scale":0.1,    #lambda parameter of process noise
                                      
                                      "resource_availability_p":0.5,    #probability of getting an agent
                                      "resource_availability_n":3,      #number of agents
                                      "resource_availability_m":0.041,  #penalty time in days
                                      "activity_duration_lambda_range":2,
                                      "Deterministic_offset_W": [[0, # CLOSED HOURS
                                                          1, 
                                                          2, 
                                                          3,
                                                          4, 
                                                          5,
                                                          6],
                                                         
                                                         [0.5, 
                                                          1.5, 
                                                          2.5, 
                                                          3.5,
                                                          4.5,
                                                          5.5,
                                                          6.5]], 
                                      "Deterministic_offset_u":7}, 
                                      custom_distribution=None,
                                      seed_value=1337):
    import numpy as np
    #np.random.seed(seed_value)
    import pandas as pd
    
    from SynBPS.simulation.Duration.duration_helpers import Generate_lambdas, Resource_offset, TimeSinceMonday, Deterministic_offset
    from SynBPS.simulation.Arrival.alg1_trace_arrivals import Generate_trace_arrivals

    
    
    """
    ############################################################
    """
    
    """
    Initialization: Generate distributions
    """    
    
    #generate ids for each case
    caseids = list(range(0,len(Theta)))
    
    # for lambdas
    max_trace_length = max(len(x) for x in Theta)
    
    # Generate duration distributions
    Lambd = Generate_lambdas(D=D, 
                             t=max_trace_length, 
                             lambd_range=settings["activity_duration_lambda_range"],
                             seed_value=seed_value)
    
    # check for custom distributions
    if custom_distribution is not None:
        # load from csv
        Lambd = pd.read_csv(custom_distribution["Lambda"])
        Lambd.columns = D

        if len(Lambd) < max_trace_length:
            raise("T-axis of Lambd is < the maximal trace length. Please specify a larger Lambda matrix.")
    
    # Generate arrival times
    theta_time, z = Generate_trace_arrivals(lambd = settings["inter_arrival_time"], 
                                            n_arrivals=len(caseids),
                                            seed_value=seed_value)
    
    # All time-information generated is stored here
    Y_table = []
    
    for idx in caseids:
        
        #get the trace from Phi
        trace = Theta[idx]
    
        #Remove absorption state from trace (duration=0)
        trace = list(filter(lambda a: a != "END", trace))
        
        # Arrival-time relative to the beginning of the timeline
        z_i = z[idx]
        
        # Matrix with boundaries
        W = settings["Deterministic_offset_W"]
        
        # Time units for a full week
        u = settings["Deterministic_offset_u"]
        
        # initialization
        y_sum = 0
        y_acc_sum = []
                
        #############################
        X = [] #activity START
        Y = [] #activity END

        B = [] #process stability offset
        H = [] #resource availability offset
        Q = [] #time since monday
        S = [] #business hours offset
        V = [] #activity duration
        Z = [] #arrival time of trace
        
        starttimes = [] #starttime in continuous time
        endtimes = [] #endtime in continuous time

        """
        ############################################################
        """
        
        for t in range(0,len(trace)):
            # Get the activity at timestep t
            e_t = trace[t]
            
            """
            ARRIVAL TIME:
                When does the case enter the system?
            """
            # if this is the first timestep in the trace, set n_t to the arrival time
            if t == 0:
                n_t = z_i # arrival time is already generated in alg 1
            
            # if not, add the duration of the previous events (Y_t-1) to the start time
            if t > 0:
                # arrivaltime + total duration of last activity incl delays
                n_t = n_t + Y[t-1]
            
            # append the values to vectors/the table
            X.append(n_t)
            Z.append(z_i)
            
            """
            RESOURCE OFFSET:
                Delays due to no resource available instantly
            """
            # sample the offset from binomial distribution: get number of trials before success x delay per trial
            h_t = Resource_offset(m = settings["resource_availability_m"], 
                                  p = settings["resource_availability_p"], 
                                  n = settings["resource_availability_n"],
                                  seed_value=seed_value)
            H.append(h_t)

            """
            PROCESS STABILITY:
                Delays due to lack of standardization
            """
            
            if settings["process_stability_scale"] == 0:
                b_t = 0
                
            if settings["process_stability_scale"] > 0:
                b_t = np.random.exponential(settings["process_stability_scale"],1)[0]
                        
            B.append(b_t)
            
            """
            DETERMINISTIC OFFSET:
                Prior to starting the work, an offset is added based on the weekday
            """
            
            # all stochastic offsets: resource + stability
            m = h_t + b_t
                                
            # get time since monday: arrivaltime or starttime + offsets
            q_t = np.mod(n_t + m, u)
            
            # Append to vector/table
            Q.append(q_t)
            
            # Get the deterministic offset
            s_t = Deterministic_offset(W, q_t)
            
            # Append to vector/table
            S.append(s_t)
            
            """
            DURATION: Event
                The duration of the work performed
            """
            
            # Hypoexponential distribution; Lookup the lambda value for activity e that t
            lambdavalue = Lambd[e_t].loc[t]
            
            # Generate the activity duration from exponential dist
            v_t = np.random.exponential(scale=lambdavalue, size=1)[0]
            V.append(v_t)
                       
            """
            TOTAL DURATION
            """
            # resource offset + stability offset + their conditional deterministic offset + activity duration
            u_t = h_t + b_t + s_t + v_t 
            
            Y.append(u_t)
            
            # accumulated time (all durations, excluding the arrival time)
            y_sum = np.sum(Y)
            y_acc_sum.append(y_sum)
            """
            TIMESTAMPS
                Continuous time variables used for the timestmaps
            """
            # if this is the first event, starttime is after the stochastic and deterministic offsets, and endtime is simply after the delay
            if t == 0:
                # arrivaltime + resource availability + stability + deterministic (calendar) offsets before work begins
                starttime_t = n_t + h_t + b_t + s_t
                endtime_t = starttime_t + v_t
            if t > 0:
                # end of last activity + resource availability + stability + deterministic (calendar) offsets before work begins
                starttime_t = (n_t + u_t) - v_t
                endtime_t = starttime_t + v_t # end of the activity is after v_t (event duration)

            starttimes.append(starttime_t)
            endtimes.append(endtime_t)

        #updated variable names to match paper (19/02)
        case_times = {"caseid":idx,

                      "z_t":Z,            # offset since beginning (arrival time of trace)

                      "n_t":X,                      # arrival times (start/ready time of activity)
                      "u_t":Y,                      # duration plus offsets (for this event only)
                      "y_acc_sum":y_acc_sum,        # accumulated durations plus offsets 
                      
                      
                      "h_t":H,                      # resource offset
                      "b_t":B,                      # stability offset
                      "q_t":Q,                      # time since monday (point in week)
                      "s_t":S,                      # calendar-based deterministic offset
                      "v_t":V,                      # event duration only

                      "starttime":starttimes,       # activity start only (after all delays)
                      "endtime":endtimes}           # activity durations only
        
        Y_table.append(case_times)
        
    return Y_table, Lambd, theta_time
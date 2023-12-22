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
                                      "Deterministic_offset_u":7}):
    
    import numpy as np
    import pandas as pd
    
    from SynBPS.simulation.duration_helpers import Generate_lambdas, Resource_offset, TimeSinceMonday, Deterministic_offset
    from SynBPS.simulation.alg1_trace_arrivals import Generate_trace_arrivals
    
    
    """
    ############################################################
    """
    
    """
    Initialization: Generate distributions etc.
    """
    
    
    #event-log 
    
    
    #generate ids for each case
    caseids = list(range(0,len(Theta)))
    
    #For testing
    max_trace_length = max(len(x) for x in Theta)
    
    # Generate duration distributions
    Lambd = Generate_lambdas(D=D, 
                             t=max_trace_length, 
                             lambd_range=settings["activity_duration_lambda_range"])
    
    # Generate arrival times
    theta_time, z = Generate_trace_arrivals(lambd = settings["inter_arrival_time"], 
                                            n_arrivals=len(caseids))
    
    
    # All time-information generated is stored here
    Y_container = []
    
    for idx in caseids:
        
        #get the trace from Phi
        trace = Theta[idx]
    
        #Remove absorption state from trace (duration=0)
        #trace.remove("END")
        trace = list(filter(lambda a: a != "END", trace))
        
        """
        Should the END state be removed here?
        """
        
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
        Z = []
        
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
            """
            if t == 0:
                n_t = z_i
                y_t = 0
                
            if t > 0:
                n_t = n_t + Y[t-1]
                
            X.append(n_t)
            Z.append(z_i)
            
            """
            RESOURCE OFFSET:
                Delays due to no resource available instantly
            """
            h_t = Resource_offset(h = 0, 
                                    m = settings["resource_availability_m"], 
                                    p = settings["resource_availability_p"], 
                                    n = settings["resource_availability_n"])
            H.append(h_t)


            """
            PROCESS STABILITY
            """
            
            if settings["process_stability_scale"] == 0:
                b_t = 0
                
            if settings["process_stability_scale"] > 0:
                b_t = np.random.exponential(settings["process_stability_scale"],1)[0]
                        
            B.append(b_t)
            


            """
            DETERMINISTIC OFFSET:
            """
            
            # all stochastic offsets
            m = h_t + b_t
                                
            # get time since monday
            q_t = np.mod(n_t+m,u)
            
            # Append to final data
            Q.append(q_t)
            
            # Get the deterministic offset
            s_t = Deterministic_offset(W, q_t)
            
            # Append to final data
            S.append(s_t)
            
            
            
            """
            DURATION
            """
            
            # Get the lambda value for that activity
            lambdavalue = Lambd[e_t].loc[t]
            
            # Generate the activity duration
            v_t = np.random.exponential(scale=lambdavalue,size=1)[0]
            V.append(v_t)
                       
            """
            TOTAL DURATION
            """

            u_t = h_t + b_t + s_t + v_t
            
            Y.append(u_t)
            
            y_sum = y_sum + y_t
            y_acc_sum.append(y_sum)

            #continuous time variables used for the timestmaps
            starttime_t = n_t + h_t + b_t + s_t 
            endtime_t = starttime_t + v_t

            starttimes.append(starttime_t)
            endtimes.append(endtime_t)

            
        #print("Trace:",trace)
        #print("Trace durations:",Y)
        #print("Acc cycle-time:",y_acc_sum)
        #print("Total cycle-time:",y_sum)
        
        #updated variable names to match paper (19/02)
        case_times = {"caseid":idx,

                      "z_t":Z,            #offset since beginning (arrival time of trace)

                      "n_t":X,                      #arrival times (start/ready time of activity)
                      "u_t":Y,                      #duration plus offsets
                      "y_acc_sum":y_acc_sum,        #accumulated durations
                      
                      
                      "h_t":H,            #resource offset
                      "b_t":B,            #stability offset
                      "q_t":Q,            #time since monday (point in week)
                      "s_t":S,            #calendar-based deterministic offset
                      "v_t":V,
                      "starttime":starttimes,
                      "endtime":endtimes}        #activity durations only
        
        Y_container.append(case_times)
        
    #### Done ####
    
    Y_container

    return Y_container, Lambd, theta_time


#Generate_time_variables(Theta = [["a","b","END"],["a","b","END"]],
#                                   D = ["a","b"])

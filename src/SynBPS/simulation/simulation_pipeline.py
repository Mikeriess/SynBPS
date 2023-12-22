#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:03:27 2021

@author: mikeriess
"""

def generate_eventlog(curr_settings, output_dir=None):

    from SynBPS.simulation.simulation_helpers import make_D, make_workweek
    
    statespace = make_D(int(curr_settings["statespace_size"]))
    number_of_traces = int(curr_settings["number_of_traces"])  
    process_entropy = curr_settings["process_entropy"] 
    process_type = curr_settings["process_type"] 
    process_memory = int(curr_settings["process_memory"]) 
    num_transitions = int(curr_settings["med_ent_n_transitions"]) 

    time_settings = {"inter_arrival_time":float(curr_settings["inter_arrival_time"]), 
                            "process_stability_scale":float(curr_settings["process_stability_scale"]),
                            "resource_availability_p":float(curr_settings["resource_availability_p"]),
                            "resource_availability_n":int(curr_settings["resource_availability_n"]),
                            "resource_availability_m":float(curr_settings["resource_availability_m"]), 
                            "activity_duration_lambda_range":float(curr_settings["activity_duration_lambda_range"]),
                            "Deterministic_offset_W":make_workweek(curr_settings["Deterministic_offset_W"]),
                            "Deterministic_offset_u":int(curr_settings["Deterministic_offset_u"])}
    
    datetime_offset = int(curr_settings["datetime_offset"])
    run = int(curr_settings["run"])
    
    
    import pandas as pd
    import numpy as np
    
    from SynBPS.simulation.alg6_memoryless_process_generator import Process_without_memory
    from SynBPS.simulation.alg7_memory_process_generator import Process_with_memory
    from SynBPS.simulation.alg9_trace_durations import Generate_time_variables
    
    """
    Simulation pipeline:
    """
    #placeholder
    #max_trace_length = 0
    
    #while loop to ensure that traces are more than just one event (which cannot be predicted from)
    #while max_trace_length < 3:

    # Generate an event-log
    if process_type == "memory":
        # HOMC only valid for medium entropy
        #if process_entropy == "med_entropy":
        #Theta, Phi = Process_with_memory(D = statespace, 
        #                        mode = process_entropy, 
        #                        num_traces=number_of_traces, 
        #                        K=process_memory)

        # HOMC not valid for min_entropy, as this is a deterministic process
        if process_entropy == "min_entropy":
            Theta, Phi = Process_without_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces,
                                num_transitions=num_transitions)
        else:
            Theta, Phi = Process_with_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces, 
                                K=process_memory)
    
    if process_type == "memoryless":
        Theta, Phi = Process_without_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces)
        
        
    # print number of traces
    print("traces:",len(Theta))
    
    # Generate time objects
    Y_container, Lambd, theta_time = Generate_time_variables(Theta = Theta,
                                                             D = statespace,
                                                             settings = time_settings)
    
    #loop over all the traces
    for i in list(range(0,len(Theta))):
        
        # get the activities
        trace = Theta[i]
        
        # remove "END" activity
        trace = list(filter(lambda a: a != "END", trace))
        
        # get the caseids
        caseids = [str(i)]*len(trace) #(max_trace_length-1)
        
        # generate timesteps
        timesteps = list(range(1,len(trace)+1))
        timesteps = [int(x) for x in timesteps]
            
        # generate a table
        trace = pd.DataFrame({"caseid":caseids,
                             "activity":trace,
                             "activity_no":timesteps,
                             "y_acc_sum":Y_container[i]["y_acc_sum"],
                             "z_t":Y_container[i]["z_t"],
                             "n_t":Y_container[i]["n_t"],
                             "q_t":Y_container[i]["q_t"],
                             "h_t":Y_container[i]["h_t"],
                             "b_t":Y_container[i]["b_t"],
                             "s_t":Y_container[i]["s_t"],
                             "v_t":Y_container[i]["v_t"],
                             "u_t":Y_container[i]["u_t"],
                             "starttime":Y_container[i]["starttime"],
                             "endtime":Y_container[i]["endtime"]})
        
        if i ==0:
            #make final table
            evlog_df = trace
    
        if i > 0:
            # append to the final table
            evlog_df = pd.concat((evlog_df,trace))
    
    # fix indexes
    evlog_df.index = list(range(0,len(evlog_df)))
    
    # convert starttime to a timestamp
    ###################################
    
    # year offset
    #year_offset = (60*60*24*365)*52
    year_offset = datetime_offset
    
    # 01/01/1970 is a thursday
    weekday_offset = 4 #+ year_offset
    
    #scaling from continuous units to preferred time unit
    time_conversion = (60*60*24)
    
    """
    Generate arrival time
    """
    evlog_df['arrival_datetime'] = (evlog_df["z_t"] + weekday_offset)*time_conversion
    evlog_df['arrival_datetime'] = evlog_df['arrival_datetime'].astype('datetime64[s]') #%yyyy-%mm-%dd %hh:%mm:%ss
        
    """
    Generate activity start time: n_t + resource availability h_t + Stability offset b_t + BH offset s_t
    """
    
    #evlog_df['start_datetime'] = ((evlog_df["Y"] - evlog_df["v_t"]) + weekday_offset)*time_conversion
    evlog_df['start_datetime'] = ((evlog_df["starttime"]) + weekday_offset)*time_conversion
    evlog_df['start_datetime'] = evlog_df['start_datetime'].astype('datetime64[s]')
    
    """
    Generate activity end time: n_t + total duration including offsets
    """
    
    evlog_df['end_datetime'] = (evlog_df["endtime"] + weekday_offset)*time_conversion
    evlog_df['end_datetime'] = evlog_df['end_datetime'].astype('datetime64[s]')
 
    # add years to dates
    evlog_df['arrival_datetime'] = evlog_df['arrival_datetime'] + pd.offsets.DateOffset(years=year_offset)
    evlog_df['start_datetime'] = evlog_df['start_datetime'] + pd.offsets.DateOffset(years=year_offset)
    evlog_df['end_datetime'] = evlog_df['end_datetime'] + pd.offsets.DateOffset(years=year_offset)

    # turn clock -6 hours back (so office hours are 06:00 - 18:00)

    evlog_df['arrival_datetime'] = evlog_df['arrival_datetime'] + pd.offsets.DateOffset(hours=-6)
    evlog_df['start_datetime'] = evlog_df['start_datetime'] + pd.offsets.DateOffset(hours=-6)
    evlog_df['end_datetime'] = evlog_df['end_datetime'] + pd.offsets.DateOffset(hours=-6)

    # turn clock -4 days back (so week starts at monday)
    evlog_df['arrival_datetime'] = evlog_df['arrival_datetime'] + pd.offsets.DateOffset(days=-3)
    evlog_df['start_datetime'] = evlog_df['start_datetime'] + pd.offsets.DateOffset(days=-3)
    evlog_df['end_datetime'] = evlog_df['end_datetime'] + pd.offsets.DateOffset(days=-3)
    

    # control: get day of week of beginning work
    evlog_df['start_day'] = evlog_df['start_datetime'].dt.day_name()
    evlog_df['start_hour'] = evlog_df['start_datetime'].apply(lambda x: x.hour)

    
    if output_dir is not None:
        evlog_df.to_csv(output_dir+str(run)+"_Eventlog_"+process_entropy+"_"+process_type+".csv", index=False)
        print("eventlog saved to:",output_dir+str(run)+"_Eventlog_"+process_entropy+"_"+process_type+".csv")
    
    print("events:",len(evlog_df))
    print("ids:",len(evlog_df.caseid.unique()))
    return evlog_df

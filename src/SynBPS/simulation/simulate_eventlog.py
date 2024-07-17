
def generate_eventlog(curr_settings, verbose=False):
    """
    Generates an event log based on specified parameters.

    Args:
        curr_settings (dict): A dictionary containing the following keys:
            number_of_traces (int): Number of traces/cases in the event log.
            process_entropy (str): Level of entropy. Options: "min_entropy", "med_entropy", "max_entropy" or "custom" for custom_distribution (see below).
            process_type (str): Type of Markov chain. Options: "memoryless", "memory".
            process_memory (int): Order of the Higher-Order Markov Chain (HOMC). Only used when process_type is "memory".
            statespace_size (int): Number of activity types.
            med_ent_n_transitions (int): Number of transitions for medium entropy. Should be > 2 and < statespace_size.
            inter_arrival_time (float): Lambda parameter of inter-arrival times.
            process_stability_scale (float): Lambda parameter of process noise.
            resource_availability_p (float): Probability of agent being available (0-1).
            resource_availability_n (int): Number of agents in the process.
            resource_availability_m (float): Waiting time in full days when no agent is available.
            activity_duration_lambda_range (float): Variation between activity durations.
            Deterministic_offset_W (str): Business hours definition. Example: "weekdays".
            Deterministic_offset_u (int): Time unit for a full week (e.g., 7 for days, 168 for hours).
            datetime_offset (int): Offset for timestamps in years after 1970.
            seed_value (int): Seed value for random number generation.
            custom_distributions(dict): (Default: None) Dictionary with filenames for custom initial probabilities, transition matrix and duration distribution. Example usage: {"p0":"data/p0.csv", "p":"data/p.csv","Lambda":"data/lambda.csv"}

    Returns:
        Pandas dataframe with the simulated event-log
    """

    # check for custom distributions
    if "custom_distributions" not in curr_settings:
        custom_dist = None
    if "custom_distributions" in curr_settings:
        print("Using custom distributions:\n", curr_settings["custom_distributions"])
        custom_dist = curr_settings["custom_distributions"]
        if curr_settings["process_entropy"] != "custom":
            raise("Custom distributions have been specified, but process entropy is set to min, med or max. Please remove custom_distributions or set process_entropy to 'custom'")

    # set the seed 
    from numpy.random import seed
    
    seed_val = int(curr_settings["seed_value"])
    if verbose==True:
        print("seed:",seed_val)
    seed(seed_val)

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
    
    import pandas as pd
    import numpy as np
    
    from SynBPS.simulation.alg6_memoryless_process_generator import Process_without_memory
    from SynBPS.simulation.alg7_memory_process_generator import Process_with_memory
    from SynBPS.simulation.Duration.alg9_trace_durations import Generate_time_variables
    
    """
    Simulation pipeline:
    """

    # Generate an event-log
    if process_type == "memory":
        if "custom_distributions" in curr_settings:
            raise Exception("Cannot use custom distribution with memory process. Change to memoryless or set custom_distributions to None")

        # HOMC not valid for min_entropy, as this is a deterministic process
        if process_entropy == "min_entropy":
            Theta, Phi = Process_without_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces,
                                num_transitions=num_transitions, 
                                seed_value=seed_val)
        else:
            Theta, Phi = Process_with_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces, 
                                K=process_memory, 
                                seed_value=seed_val)
    
    if process_type == "memoryless":
        Theta, Phi = Process_without_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces, 
                                custom_distribution=custom_dist,
                                seed_value=seed_val)
        
        
    # print number of traces
    if verbose==True:
        print("traces:",len(Theta))
    
    # Generate time objects
    Y_container, Lambd, theta_time = Generate_time_variables(Theta = Theta,
                                                            D = statespace,
                                                            settings = time_settings, 
                                                            custom_distribution=custom_dist,
                                                            seed_value=seed_val)
    
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
    
    if verbose==True:
        print("events:",len(evlog_df))
        print("ids:",len(evlog_df.caseid.unique()))
    return evlog_df

###########
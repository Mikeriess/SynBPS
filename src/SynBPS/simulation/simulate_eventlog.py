def check_settings(curr_settings):
    """
    Checks the settings of an event-log before the simulation starts.

    Args:
        curr_settings (dict): The settings dictionary, or a row of a design table. See generate_eventlog for the keys.

    Raises:
        ValueError: One line per problem found, each ending with the setting to change.
    """

    # problems found so far
    problems = []

    # settings which every event-log needs
    required = ["number_of_traces", "process_entropy", "process_type", "process_memory", "statespace_size",
                "med_ent_n_transitions", "inter_arrival_time", "process_stability_scale", "resource_availability_p",
                "resource_availability_n", "resource_availability_m", "activity_duration_lambda_range",
                "Deterministic_offset_W", "Deterministic_offset_u", "datetime_offset", "seed_value"]

    for key in required:
        if key not in curr_settings:
            problems.append(key + " is missing. Add " + key + " to the settings.")

    # a setting as a number, or None if it is missing or not a number (which is noted as a problem)
    def number(key):
        if key not in curr_settings:
            return None
        try:
            return float(curr_settings[key])
        except (TypeError, ValueError):
            problems.append(key + " must be a number, but is " + str(curr_settings[key]) + ". Change " + key + ".")
            return None

    # a range check, written such that a missing value (NaN) also fails
    def in_range(value, low, high, low_included=True, high_included=True):
        if low_included and high_included:
            return low <= value <= high
        if low_included and not high_included:
            return low <= value < high
        if not low_included and high_included:
            return low < value <= high
        return low < value < high

    process_type = curr_settings["process_type"] if "process_type" in curr_settings else None
    process_entropy = curr_settings["process_entropy"] if "process_entropy" in curr_settings else None

    if process_type is not None and process_type not in ["memoryless", "memory"]:
        problems.append("process_type must be memoryless or memory, but is " + str(process_type) + ". Change process_type.")

    if process_entropy is not None and process_entropy not in ["min_entropy", "med_entropy", "max_entropy", "custom"]:
        problems.append("process_entropy must be min_entropy, med_entropy, max_entropy or custom, but is " + str(process_entropy) + ". Change process_entropy.")

    # custom distributions: only for the memoryless process, and only with process_entropy custom
    custom_given = "custom_distributions" in curr_settings and curr_settings["custom_distributions"] is not None

    if process_entropy == "custom" and not custom_given:
        problems.append("process_entropy is custom, but no custom_distributions were given. Add custom_distributions with the files p0, p and Lambda, or change process_entropy.")

    if custom_given and process_entropy != "custom":
        problems.append("custom_distributions were given, but process_entropy is not custom. Remove custom_distributions or set process_entropy to custom.")

    if custom_given and process_type == "memory":
        problems.append("custom_distributions cannot be used with the process with memory. Change process_type to memoryless or remove custom_distributions.")

    number_of_traces = number("number_of_traces")
    if number_of_traces is not None and not in_range(number_of_traces, 1, float("inf")):
        problems.append("number_of_traces must be 1 or larger, but is " + str(number_of_traces) + ". Change number_of_traces.")

    # the alphabet in make_D has room for 27 activities
    statespace_size = number("statespace_size")
    if statespace_size is not None and not in_range(statespace_size, 2, 27):
        problems.append("statespace_size must be between 2 and 27, but is " + str(statespace_size) + ". Change statespace_size.")

    # number of transitions: only used for medium entropy. the process with memory allows the absorption state as one of them
    num_transitions = number("med_ent_n_transitions")
    if process_entropy == "med_entropy" and num_transitions is not None and statespace_size is not None:
        upper = statespace_size + 1 if process_type == "memory" else statespace_size
        if not in_range(num_transitions, 2, upper):
            problems.append("med_ent_n_transitions must be between 2 and " + str(int(upper)) + " for this statespace_size and process_type, but is " + str(num_transitions) + ". Change med_ent_n_transitions.")

    # settings which are only used by the process with memory
    if process_type == "memory":
        process_memory = number("process_memory")
        if process_memory is not None and not in_range(process_memory, 1, float("inf")):
            problems.append("process_memory must be 1 or larger for the process with memory, but is " + str(process_memory) + ". Change process_memory.")

        p_abs_min = number("p_abs_min")
        if p_abs_min is not None and not in_range(p_abs_min, 0, 1, high_included=False):
            problems.append("p_abs_min must be between 0 and 1 (1 excluded), but is " + str(p_abs_min) + ". Change p_abs_min.")

        max_len = number("max_len")
        if max_len is not None and not in_range(max_len, 2, float("inf")):
            problems.append("max_len must be 2 or larger, but is " + str(max_len) + ". Change max_len.")

    # time-related settings
    p = number("resource_availability_p")
    if p is not None and not in_range(p, 0, 1, low_included=False):
        problems.append("resource_availability_p must be between 0 (excluded) and 1, but is " + str(p) + ". Change resource_availability_p.")

    n = number("resource_availability_n")
    if n is not None and not in_range(n, 1, float("inf")):
        problems.append("resource_availability_n must be 1 or larger, but is " + str(n) + ". Change resource_availability_n.")

    m = number("resource_availability_m")
    if m is not None and not in_range(m, 0, float("inf")):
        problems.append("resource_availability_m must be 0 or larger, but is " + str(m) + ". Change resource_availability_m.")

    stability = number("process_stability_scale")
    if stability is not None and not in_range(stability, 0, float("inf")):
        problems.append("process_stability_scale must be 0 or larger, but is " + str(stability) + ". Change process_stability_scale.")

    inter_arrival = number("inter_arrival_time")
    if inter_arrival is not None and not in_range(inter_arrival, 0, float("inf"), low_included=False):
        problems.append("inter_arrival_time must be larger than 0, but is " + str(inter_arrival) + ". Change inter_arrival_time.")

    # the durations are drawn from Uniform(0.0001, activity_duration_lambda_range)
    lambd_range = number("activity_duration_lambda_range")
    if lambd_range is not None and not in_range(lambd_range, 0.0001, float("inf"), low_included=False):
        problems.append("activity_duration_lambda_range must be larger than 0.0001, but is " + str(lambd_range) + ". Change activity_duration_lambda_range.")

    u = number("Deterministic_offset_u")
    if u is not None and not in_range(u, 0, float("inf"), low_included=False):
        problems.append("Deterministic_offset_u must be larger than 0, but is " + str(u) + ". Change Deterministic_offset_u.")

    if "Deterministic_offset_W" in curr_settings and curr_settings["Deterministic_offset_W"] not in ["weekdays", "all-week", "none"]:
        problems.append("Deterministic_offset_W must be weekdays, all-week or none, but is " + str(curr_settings["Deterministic_offset_W"]) + ". Change Deterministic_offset_W.")

    # numpy accepts seeds from 0 to 2**32 - 1
    seed_value = number("seed_value")
    if seed_value is not None and not in_range(seed_value, 0, 2**32 - 1):
        problems.append("seed_value must be between 0 and 2**32 - 1, but is " + str(seed_value) + ". Change seed_value.")

    if len(problems) > 0:
        raise ValueError("Invalid settings for the event-log:\n" + "\n".join(problems))


def generate_eventlog(curr_settings, verbose=False):
    """
    Generates an event log based on specified parameters.

    Args:
        curr_settings (dict): A dictionary containing the following keys:
            number_of_traces (int): Number of traces/cases in the event log.
            process_entropy (str): Level of entropy. Options: "min_entropy", "med_entropy", "max_entropy" or "custom" for custom_distribution (see below).
            process_type (str): Type of Markov chain. Options: "memoryless", "memory".
            process_memory (int): Order of the Higher-Order Markov Chain (HOMC). Only used when process_type is "memory".
            p_abs_min (float): (Default: 0.05) Minimum probability of ending the trace from any state. Only used when process_type is "memory" and process_entropy is "med_entropy" or "max_entropy". A value of 0 removes the guarantee that every trace ends.
            max_len (int): (Default: 10000) Maximum number of events in a trace, after which an exception is raised. Only used when process_type is "memory".
            statespace_size (int): Number of activity types.
            med_ent_n_transitions (int): Number of possible transitions from each state. Only used when process_entropy is "med_entropy". Between 2 and statespace_size (statespace_size + 1 for the process with memory, where the absorption state can be one of them).
            inter_arrival_time (float): Lambda parameter of inter-arrival times.
            process_stability_scale (float): Lambda parameter of process noise.
            resource_availability_p (float): Probability of agent being available (0-1).
            resource_availability_n (int): Number of agents in the process.
            resource_availability_m (float): Waiting time in full days when no agent is available.
            activity_duration_lambda_range (float): Variation between activity durations.
            Deterministic_offset_W (str): Business hours definition. Options: "weekdays", "all-week" or "none" (no closed hours).
            Deterministic_offset_u (int): Time unit for a full week (e.g., 7 for days, 168 for hours).
            datetime_offset (int): Offset for timestamps in years after 1970.
            seed_value (int): Seed value for random number generation.
            custom_distributions(dict): (Default: None) Dictionary with filenames for custom initial probabilities, transition matrix and duration distribution. Example usage: {"p0":"data/p0.csv", "p":"data/p.csv","Lambda":"data/lambda.csv"}

    Returns:
        Pandas dataframe with the simulated event-log

    Raises:
        ValueError: If a setting is missing or invalid. The message lists every problem found.
    """

    # check all settings first
    check_settings(curr_settings)

    # check for custom distributions (the key may be missing or None)
    custom_dist = None
    if "custom_distributions" in curr_settings and curr_settings["custom_distributions"] is not None:
        print("Using custom distributions:\n", curr_settings["custom_distributions"])
        custom_dist = curr_settings["custom_distributions"]

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

    # minimum probability of absorption from any state (default 0.05) - only used for process with memory
    p_abs_min = float(curr_settings["p_abs_min"]) if "p_abs_min" in curr_settings else 0.05

    # maximum number of events in a trace (default 10000) - only used for process with memory
    max_len = int(curr_settings["max_len"]) if "max_len" in curr_settings else 10000

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
        # HOMC of order K for all entropy levels (min_entropy is a deterministic process with memory)
        Theta, Phi = Process_with_memory(D = statespace, 
                            mode = process_entropy, 
                            num_traces=number_of_traces, 
                            K=process_memory, 
                            num_transitions=num_transitions, 
                            p_abs_min=p_abs_min, 
                            max_len=max_len, 
                            seed_value=seed_val)
    
    if process_type == "memoryless":
        Theta, Phi = Process_without_memory(D = statespace, 
                                mode = process_entropy, 
                                num_traces=number_of_traces, 
                                num_transitions=num_transitions, 
                                custom_distribution=custom_dist,
                                seed_value=seed_val)
        
        
    # print number of traces
    if verbose==True:
        print("traces:",len(Theta))
    
    # Generate time objects
    Y_table, Lambd, theta_time = Generate_time_variables(Theta = Theta,
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
                             "y_acc_sum":Y_table[i]["y_acc_sum"],
                             "z_t":Y_table[i]["z_t"],
                             "n_t":Y_table[i]["n_t"],
                             "q_t":Y_table[i]["q_t"],
                             "h_t":Y_table[i]["h_t"],
                             "b_t":Y_table[i]["b_t"],
                             "s_t":Y_table[i]["s_t"],
                             "v_t":Y_table[i]["v_t"],
                             "u_t":Y_table[i]["u_t"],
                             "starttime":Y_table[i]["starttime"],
                             "endtime":Y_table[i]["endtime"]})
        
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
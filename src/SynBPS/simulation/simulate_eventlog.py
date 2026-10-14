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

    # the alphabet in make_D has room for 29 activities
    statespace_size = number("statespace_size")
    if statespace_size is not None and not in_range(statespace_size, 2, 29):
        problems.append("statespace_size must be between 2 and 29, but is " + str(statespace_size) + ". Change statespace_size.")

    # number of transitions: only used for medium entropy. the process with memory allows the absorption state as one of them
    num_transitions = number("med_ent_n_transitions")
    if process_entropy == "med_entropy" and num_transitions is not None and statespace_size is not None:
        upper = statespace_size + 1 if process_type == "memory" else statespace_size
        if not in_range(num_transitions, 2, upper):
            problems.append("med_ent_n_transitions must be between 2 and " + str(int(upper)) + " for this statespace_size and process_type, but is " + str(num_transitions) + ". Change med_ent_n_transitions.")

    # kappa: only used by the memoryless process with medium entropy. the sum of the absorption probabilities
    # is at least 1 (the absorption state), and the search in alg. 5 becomes very slow above 1 + statespace_size/2
    kappa = number("med_ent_kappa")
    if process_entropy == "med_entropy" and process_type == "memoryless" and kappa is not None and statespace_size is not None:
        if not in_range(kappa, 0, 1 + statespace_size/2):
            problems.append("med_ent_kappa must be between 0 and 1 + statespace_size/2 (" + str(1 + statespace_size/2) + ") for this statespace_size, but is " + str(kappa) + ". Change med_ent_kappa.")

    # settings which are only used by the process with memory
    if process_type == "memory":
        process_memory = number("process_memory")
        if process_memory is not None and not in_range(process_memory, 1, float("inf")):
            problems.append("process_memory must be 1 or larger for the process with memory, but is " + str(process_memory) + ". Change process_memory.")

        p_abs_min = number("p_abs_min")
        if p_abs_min is not None and not in_range(p_abs_min, 0, 1, high_included=False):
            problems.append("p_abs_min must be between 0 and 1 (1 excluded), but is " + str(p_abs_min) + ". Change p_abs_min.")

    # the maximum trace length is used by both process types
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

    if "resource_availability_mode" in curr_settings and curr_settings["resource_availability_mode"] not in ["all_requests", "failed_requests"]:
        problems.append("resource_availability_mode must be all_requests or failed_requests, but is " + str(curr_settings["resource_availability_mode"]) + ". Change resource_availability_mode.")

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


def timestamp_origin(datetime_offset):
    """
    The timestamp of continuous time t = 0: the first Sunday of the year 1970 + datetime_offset, at 18:00.

    In continuous time, the week starts Monday 00:00 and the closed hours of the business-hour presets are
    00:00 to 12:00 of each day. The timestamps are placed 6 hours earlier, so the open hours are 06:00 to 18:00,
    Monday to Friday, for every datetime_offset.

    Args:
        datetime_offset (int): Years added to 1970.

    Returns:
        pandas Timestamp
    """
    import pandas as pd

    jan_1 = pd.Timestamp(year=1970 + int(datetime_offset), month=1, day=1)

    # days from January 1st to the first Sunday (weekday 6)
    days_to_sunday = (6 - jan_1.weekday()) % 7

    return jan_1 + pd.Timedelta(days=days_to_sunday, hours=18)


def continuous_to_datetime(days, origin):
    """
    Converts continuous time, in days since the start of the simulation, to timestamps.

    Args:
        days (pandas Series): Points in continuous time, in days.
        origin (pandas Timestamp): The timestamp of t = 0, see timestamp_origin.

    Returns:
        pandas Series of timestamps, with whole seconds.
    """
    import numpy as np
    import pandas as pd

    # whole seconds. Rounded, so that an activity which waited for the opening hours starts at 06:00:00
    # (truncation gave 05:59:59 when the product is a fraction of a second below a whole second)
    seconds = np.round(days * (60*60*24))

    return origin + pd.to_timedelta(seconds, unit="s")


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
            max_len (int): (Default: 10000) Maximum number of events in a trace, after which an exception is raised.
            statespace_size (int): Number of activity types (2 to 29). The activities are named with the letters a, b, c, ...
            med_ent_n_transitions (int): Number of possible transitions from each state, drawn from the activities and the absorption state. Only used when process_entropy is "med_entropy". Between 2 and statespace_size (statespace_size + 1 for the process with memory).
            med_ent_kappa (float): (Default: 2) Minimum of the total probability of ending the trace, summed over all states, where the absorption state contributes 1 (kappa in alg. 5). States get random transitions to the absorption state until the sum is reached, so a higher value gives shorter traces. Between 0 and 1 + statespace_size/2. Only used when process_type is "memoryless" and process_entropy is "med_entropy".
            inter_arrival_time (float): Mean time between the arrivals of two traces, in days (the scale of the exponential distribution, not a rate).
            process_stability_scale (float): Mean of the exponential noise added before each event, in days. 0 means no noise.
            resource_availability_p (float): Probability that an agent is available at a request (above 0, at most 1).
            resource_availability_n (int): Number of agents in the process.
            resource_availability_m (float): Time per request until an agent is available, in days. 0.041 days is about one hour.
            resource_availability_mode (str): (Default: "all_requests") Which requests take the time resource_availability_m. "all_requests": every request, also the successful one, so the offset is at least m. "failed_requests": only the requests at which no agent was available, so the offset is 0 when an agent is available at the first request (alg. 8).
            activity_duration_lambda_range (float): Upper limit of the uniform distribution (from 0.0001) the mean activity durations Lambda(d, t) are drawn from, in days. The duration of an event is exponential with that mean.
            Deterministic_offset_W (str): Business hours definition. Options: "weekdays", "all-week" or "none" (no closed hours).
            Deterministic_offset_u (int): Length of a week in the time unit used (7 for days). The business hours of Deterministic_offset_W assume 7.
            datetime_offset (int): Offset for timestamps in years after 1970. The simulation starts (t = 0) on the first Sunday of that year at 18:00, so the business hours fall on the same weekdays for every offset.
            seed_value (int): Seed of all random streams (0 to 2**32 - 1). The same settings and seed give the same event-log.
            custom_distributions(dict): (Default: None) Dictionary with filenames for custom initial probabilities, transition matrix and duration distribution. Example usage: {"p0":"data/p0.csv", "p":"data/p.csv","Lambda":"data/lambda.csv"}

        All exponential parameters (inter_arrival_time, process_stability_scale, Lambda(d, t)) are means in days, the scale of numpy.

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

    # the seed of every random stream of the simulation (see make_rng). The global numpy stream is not used
    seed_val = int(curr_settings["seed_value"])
    if verbose==True:
        print("seed:",seed_val)

    from SynBPS.simulation.simulation_helpers import make_D, make_workweek
    
    statespace = make_D(int(curr_settings["statespace_size"]))
    number_of_traces = int(curr_settings["number_of_traces"])  
    process_entropy = curr_settings["process_entropy"] 
    process_type = curr_settings["process_type"] 
    process_memory = int(curr_settings["process_memory"]) 
    num_transitions = int(curr_settings["med_ent_n_transitions"]) 

    # minimum probability of absorption from any state (default 0.05) - only used for process with memory
    p_abs_min = float(curr_settings["p_abs_min"]) if "p_abs_min" in curr_settings else 0.05

    # minimum total probability of absorption, kappa in alg. 5 (default 2) - only used for the memoryless process with medium entropy
    kappa = float(curr_settings["med_ent_kappa"]) if "med_ent_kappa" in curr_settings else 2.0

    # maximum number of events in a trace (default 10000)
    max_len = int(curr_settings["max_len"]) if "max_len" in curr_settings else 10000

    time_settings = {"inter_arrival_time":float(curr_settings["inter_arrival_time"]), 
                    "process_stability_scale":float(curr_settings["process_stability_scale"]),
                    "resource_availability_p":float(curr_settings["resource_availability_p"]),
                    "resource_availability_n":int(curr_settings["resource_availability_n"]),
                    "resource_availability_m":float(curr_settings["resource_availability_m"]), 
                    "resource_availability_mode":curr_settings["resource_availability_mode"] if "resource_availability_mode" in curr_settings else "all_requests",
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
                                kappa=kappa,
                                custom_distribution=custom_dist,
                                max_len=max_len,
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
    
    # convert continuous time to timestamps
    ###################################

    # In continuous time, t = 0 is Monday 00:00, and the closed hours of the presets in make_workweek
    # are the first half of each day (00:00 to 12:00). The timestamps are placed 6 hours earlier, so the
    # open hours are 06:00 to 18:00: t = 0 is a Sunday at 18:00, in the year 1970 + datetime_offset.
    # The first Sunday of that year is used, so the log starts within the year and the weekdays are the
    # same for every datetime_offset.
    origin = timestamp_origin(datetime_offset)

    evlog_df['arrival_datetime'] = continuous_to_datetime(evlog_df["z_t"], origin)
    evlog_df['start_datetime'] = continuous_to_datetime(evlog_df["starttime"], origin)
    evlog_df['end_datetime'] = continuous_to_datetime(evlog_df["endtime"], origin)


    # control: get day of week of beginning work
    evlog_df['start_day'] = evlog_df['start_datetime'].dt.day_name()
    evlog_df['start_hour'] = evlog_df['start_datetime'].apply(lambda x: x.hour)
    
    if verbose==True:
        print("events:",len(evlog_df))
        print("ids:",len(evlog_df.caseid.unique()))
    return evlog_df

###########
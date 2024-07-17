import pytest

def test_basic_simulation():
    # Create a simple process
    eventlog_settings = {
                    # number of traces/cases in the event-log
                    "number_of_traces":100,

                    # level of entropy: min, medium and/or max
                    "process_entropy": "med_entropy",#"min_entropy","med_entropy","max_entropy"

                    # first or higher-order markov chain to represent the transitions "memoryless", "memory"
                    "process_type":"memory",#"memoryless",

                    # order of the HOMC - only specify this when using process with memory
                    "process_memory":2,

                    # number of activity types
                    "statespace_size":5,
                    
                    # number of transitions - only used for medium entropy (should be higher than 2 and < statespace size)
                    "med_ent_n_transitions":3,
                                    
                    # lambda parameter of inter-arrival times
                    "inter_arrival_time":1.5,
                    
                    # lambda parameter of process noise
                    "process_stability_scale":0.1,
                    
                    # probability of agent being available
                    "resource_availability_p":0.5,

                    # number of agents in the process
                    "resource_availability_n":3,

                    # waiting time in full days, when no agent is available. 0.041 is 15 minutes
                    "resource_availability_m":0.041,
                    
                    # variation between activity durations
                    "activity_duration_lambda_range":1,
                    
                    # business hours definition: when can cases be processed? ()
                    "Deterministic_offset_W":"weekdays",

                    # time-unit for a full week: days = 7, hrs = 24*7, etc.
                    "Deterministic_offset_u":7,

                    # offset for the timestamps used (years after 1970)
                    "datetime_offset":54,

                    # seed value for replication: Set this to a fixed value if the results should be reproducible
                    #"seed_value":int(np.random.uniform(low=0, high=2**32 - 1))
                    "seed_value":1337
                    }

    from SynBPS.simulation.simulate_eventlog import generate_eventlog

    log = generate_eventlog(eventlog_settings, verbose=True)

    assert len(log) > 100
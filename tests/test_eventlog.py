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

                    # minimum probability of ending the trace from any state - only used for process with memory
                    "p_abs_min":0.05,

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

def eventlog_settings(**overrides):
    """
    Settings for a small process with memory, used by the tests below.
    Any setting can be overwritten with a keyword argument.
    """
    settings = {
                    # number of traces/cases in the event-log
                    "number_of_traces":100,

                    # level of entropy: min, medium and/or max
                    "process_entropy": "med_entropy",#"min_entropy","med_entropy","max_entropy"

                    # first or higher-order markov chain to represent the transitions "memoryless", "memory"
                    "process_type":"memory",#"memoryless",

                    # order of the HOMC - only specify this when using process with memory
                    "process_memory":2,

                    # minimum probability of ending the trace from any state - only used for process with memory
                    "p_abs_min":0.05,

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
                    "seed_value":1337
                    }
    
    #overwrite the settings given as keyword arguments
    settings.update(overrides)
    
    return settings


def test_memory_orders():
    """
    Every order K and entropy level of the process with memory must produce an event-log
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog

    for process_entropy in ["min_entropy","med_entropy","max_entropy"]:
        for K in [1,2,3,4,5]:
            log = generate_eventlog(eventlog_settings(process_entropy=process_entropy, 
                                                      process_memory=K, 
                                                      number_of_traces=30))
            
            #every trace has at least one event, as a trace cannot start in the absorption state
            assert len(log.caseid.unique()) == 30
            assert len(log) >= 30

    #p_abs_min has a default value, when the setting is not given
    settings = eventlog_settings(process_memory=2)
    del settings["p_abs_min"]
    log = generate_eventlog(settings)
    assert len(log.caseid.unique()) == 100


def test_memory_context_probabilities():
    """
    One table per order 1..K, one row per context, and every row sums to 1
    """
    from SynBPS.simulation.alg7_memory_process_generator import create_homc
    
    D = ["a","b","c","d","e"]
    
    for mode in ["min_entropy","med_entropy","max_entropy"]:
        for K in [1,2,3,4]:
            HOMC = create_homc(D, K=K, mode=mode, n_transitions=3, p_abs_min=0.05, seed_value=K)
            Phi = HOMC["Phi"]
            
            #one table per order
            assert sorted(Phi.keys()) == list(range(1,K+1))
            
            for i in range(1,K+1):
                #one row per context of length i
                assert len(Phi[i]) == len(D)**i
                
                for context, row in Phi[i].items():
                    assert len(context) == i
                    assert len(row) == len(D)+1
                    assert abs(sum(row) - 1) < 1e-9
                    
                    #the absorption state has at least probability p_abs_min in every context
                    if mode != "min_entropy":
                        assert row[-1] >= 0.05 - 1e-12
            
            #P0 is a distribution over D only
            assert len(HOMC["P0"]) == len(D)
            assert abs(sum(HOMC["P0"]) - 1) < 1e-9


def test_memory_effective_order():
    """
    For K = 2, the next activity depends on the activity two steps back (given the last activity),
    and the sampled traces follow the order 2 table
    """
    import numpy as np
    from SynBPS.simulation.alg7_memory_process_generator import Process_with_memory
    
    D = ["a","b","c","d","e"]
    Theta, HOMC = Process_with_memory(D=D, mode="med_entropy", num_traces=3000, K=2, num_transitions=3, p_abs_min=0.05, seed_value=1)
    D_abs = HOMC["D_abs"]
    Phi = HOMC["Phi"]
    
    #count the transitions (e_t-2, e_t-1) -> e_t
    counts = {}
    for Q in Theta:
        for t in range(2,len(Q)):
            context = (Q[t-2], Q[t-1])
            if context not in counts:
                counts[context] = np.zeros(len(D_abs))
            counts[context][D_abs.index(Q[t])] += 1
    
    #empirical conditional distributions of the contexts with enough observations
    empirical = {context: c/np.sum(c) for context, c in counts.items() if np.sum(c) >= 200}
    assert len(empirical) >= 5
    
    #the empirical distributions must match the order 2 table (total variation distance)
    max_dev = max(0.5*np.sum(np.abs(emp - Phi[2][context])) for context, emp in empirical.items())
    assert max_dev < 0.1
    
    #and they must differ between contexts, which share the last activity
    max_diff = 0
    for last in D:
        rows = [emp for context, emp in empirical.items() if context[1] == last]
        for r1 in rows:
            for r2 in rows:
                max_diff = max(max_diff, 0.5*np.sum(np.abs(r1 - r2)))
    assert max_diff > 0.3


def test_memory_absorption():
    """
    Every trace reaches the absorption state, for a range of seeds
    """
    from SynBPS.simulation.alg7_memory_process_generator import Process_with_memory
    
    D = ["a","b","c","d","e"]
    
    for seed in range(0,100):
        Theta, HOMC = Process_with_memory(D=D, mode="med_entropy", num_traces=10, K=2, num_transitions=5, seed_value=seed)
        assert len(Theta) == 10
        
        for Q in Theta:
            #the absorption state is the last event, and only the last event
            assert Q[-1] == "END"
            assert "END" not in Q[:-1]
            assert len(Q) > 1


def test_memory_n_transitions():
    """
    med_ent_n_transitions determines the number of possible transitions from each context
    """
    import numpy as np
    from SynBPS.simulation.alg7_memory_process_generator import create_homc
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    D = ["a","b","c","d","e"]
    
    #without the absorption guarantee, exactly n transitions per context
    for n in [2,3,6]:
        HOMC = create_homc(D, K=2, mode="med_entropy", n_transitions=n, p_abs_min=0, seed_value=1)
        for i in [1,2]:
            for row in HOMC["Phi"][i].values():
                assert np.sum(row > 0) == n
    
    #with the absorption guarantee, the absorption state can be added to the transitions
    HOMC = create_homc(D, K=2, mode="med_entropy", n_transitions=2, p_abs_min=0.05, seed_value=1)
    for row in HOMC["Phi"][2].values():
        assert 2 <= np.sum(row > 0) <= 3
    
    #the setting reaches the process through generate_eventlog: at most n activities follow an activity
    log = generate_eventlog(eventlog_settings(process_memory=1, med_ent_n_transitions=2, number_of_traces=200))
    successors = {}
    for caseid in log.caseid.unique():
        activities = log.loc[log.caseid == caseid].sort_values("activity_no").activity.tolist()
        for t in range(1,len(activities)):
            successors.setdefault(activities[t-1], set()).add(activities[t])
    assert max(len(s) for s in successors.values()) <= 2


def test_memory_seed_reproducibility():
    """
    The same seed gives the same control-flow, a different seed gives a different control-flow
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    log1 = generate_eventlog(eventlog_settings(process_memory=3, seed_value=42))
    log2 = generate_eventlog(eventlog_settings(process_memory=3, seed_value=42))
    log3 = generate_eventlog(eventlog_settings(process_memory=3, seed_value=43))
    
    assert log1.caseid.tolist() == log2.caseid.tolist()
    assert log1.activity.tolist() == log2.activity.tolist()
    assert log1.activity.tolist() != log3.activity.tolist()


def test_memory_min_entropy():
    """
    Minimum entropy with memory is one deterministic trace
    """
    from SynBPS.simulation.alg7_memory_process_generator import Process_with_memory
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    D = ["a","b","c","d","e"]
    
    #K = 1: the trace is a permutation of D, as in alg 3
    Theta, HOMC = Process_with_memory(D=D, mode="min_entropy", num_traces=20, K=1, seed_value=7)
    assert len(set(tuple(Q) for Q in Theta)) == 1
    assert Theta[0][-1] == "END"
    assert sorted(Theta[0][:-1]) == sorted(D)
    
    #K = 2: one deterministic trace of the same length, in which activities may repeat
    for seed in range(0,20):
        Theta, HOMC = Process_with_memory(D=D, mode="min_entropy", num_traces=20, K=2, seed_value=seed)
        assert len(set(tuple(Q) for Q in Theta)) == 1
        assert len(Theta[0]) == len(D)+1
    
    #one variant in the event-log
    log = generate_eventlog(eventlog_settings(process_entropy="min_entropy", process_memory=2))
    variants = set()
    for caseid in log.caseid.unique():
        variants.add(tuple(log.loc[log.caseid == caseid].sort_values("activity_no").activity.tolist()))
    assert len(variants) == 1


def test_memory_invalid_settings():
    """
    Invalid settings raise an exception instead of an endless loop or a silent default
    """
    from SynBPS.simulation.alg7_memory_process_generator import create_homc
    
    D = ["a","b","c","d","e"]
    
    with pytest.raises(Exception):
        create_homc(D, K=0)
    
    with pytest.raises(Exception):
        create_homc(D, K=2, mode="med_entropy", n_transitions=1)
    
    with pytest.raises(Exception):
        create_homc(D, K=2, mode="med_entropy", n_transitions=len(D)+2)
    
    with pytest.raises(Exception):
        create_homc(D, K=2, p_abs_min=1)
    
    with pytest.raises(Exception):
        create_homc(D, K=2, mode="custom")

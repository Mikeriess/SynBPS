import pytest

from test_eventlog import eventlog_settings

def test_basic_simulation():
    # Create a simple process
    eventlog_settings = {
                    # number of traces/cases in the event-log
                    "number_of_traces":10,

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

    # create the first log using seed
    log = generate_eventlog(eventlog_settings, verbose=True)

    # create the second log using seed
    log2 = generate_eventlog(eventlog_settings, verbose=True)

    # the logs must be identical in every column
    import pandas as pd
    pd.testing.assert_frame_equal(log, log2)

def test_memoryless_process_uses_its_seed():
    """
    Direct calls of the memoryless process are reproducible from seed_value, whatever the global state was before
    """
    import numpy as np
    from SynBPS.simulation.alg6_memoryless_process_generator import Process_without_memory
    
    D = ["a","b","c","d","e"]
    
    np.random.seed(0)
    Theta1, Phi1 = Process_without_memory(D=D, mode="med_entropy", num_traces=20, num_transitions=3, seed_value=9)
    
    np.random.seed(1)
    Theta2, Phi2 = Process_without_memory(D=D, mode="med_entropy", num_traces=20, num_transitions=3, seed_value=9)
    
    assert Theta1 == Theta2
    assert Phi1[1].equals(Phi2[1])


def test_make_rng_streams():
    """
    One independent random stream per component, reproducible from the seed
    """
    import numpy as np
    from SynBPS.simulation.simulation_helpers import make_rng
    
    #same seed and stream: same draws
    assert np.array_equal(make_rng(5, "lambdas").random(5), make_rng(5, "lambdas").random(5))
    
    #different streams of the same seed differ
    assert not np.array_equal(make_rng(5, "lambdas").random(5), make_rng(5, "arrivals").random(5))
    
    #the sampling stream of seed 5 is not the table stream of seed 6, and no stream is the plain generator of the seed
    assert not np.array_equal(make_rng(5, "homc_sampling").random(5), make_rng(6, "homc_tables").random(5))
    assert not np.array_equal(make_rng(5, "homc_tables").random(5), np.random.default_rng(5).random(5))
    
    #unknown stream
    with pytest.raises(ValueError):
        make_rng(5, "durations")


def test_lambda_rows_stable():
    """
    The duration parameters Lambda depend on the seed only, and the first rows do not change when the number of timesteps grows
    """
    import numpy as np
    import pandas as pd
    from SynBPS.simulation.Duration.duration_helpers import Generate_lambdas
    
    D = ["a","b","c","d","e"]
    
    small = Generate_lambdas(D, t=3, lambd_range=1, seed_value=5)
    big = Generate_lambdas(D, t=8, lambd_range=1, seed_value=5)
    
    #timesteps are rows, activities are columns
    assert big.shape == (8, 5)
    assert list(big.columns) == D
    
    #the first rows are the same for a longer matrix
    pd.testing.assert_frame_equal(small, big.iloc[:3])
    
    #values are in the range of the uniform distribution
    assert ((big.values >= 0.0001) & (big.values < 1)).all()
    
    #another seed gives other values
    other = Generate_lambdas(D, t=8, lambd_range=1, seed_value=6)
    assert not big.equals(other)
    
    #the global numpy stream is not used
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    Generate_lambdas(D, t=8, lambd_range=1, seed_value=5)
    after = np.random.random()
    assert before == after


def test_arrival_times_stable():
    """
    The arrival times depend on the seed only, and the first arrivals do not change when more traces are generated
    """
    import numpy as np
    from SynBPS.simulation.Arrival.alg1_trace_arrivals import Generate_trace_arrivals
    
    theta_small, z_small = Generate_trace_arrivals(lambd=1.5, n_arrivals=5, seed_value=5)
    theta_big, z_big = Generate_trace_arrivals(lambd=1.5, n_arrivals=9, seed_value=5)
    assert z_small == z_big[:5]
    assert len(z_big) == 9
    
    #another seed gives other arrival times
    theta_other, z_other = Generate_trace_arrivals(lambd=1.5, n_arrivals=9, seed_value=6)
    assert z_other != z_big
    
    #the global numpy stream is not used
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    Generate_trace_arrivals(lambd=1.5, n_arrivals=9, seed_value=5)
    after = np.random.random()
    assert before == after


def test_resource_offset():
    """
    The resource offset is a multiple of m, reproducible from its stream, and has the mean of the geometric waiting time
    """
    import numpy as np
    from SynBPS.simulation.simulation_helpers import make_rng
    from SynBPS.simulation.Duration.duration_helpers import Resource_offset
    
    #same stream, same waiting times
    rng1 = make_rng(1, "resource")
    rng2 = make_rng(1, "resource")
    h1 = [Resource_offset(m=0.041, p=0.5, n=3, rng=rng1) for i in range(20)]
    h2 = [Resource_offset(m=0.041, p=0.5, n=3, rng=rng2) for i in range(20)]
    assert h1 == h2
    
    #every waiting time is a positive multiple of m
    for h in h1:
        assert h >= 0.041
        assert abs(h/0.041 - round(h/0.041)) < 1e-9
    
    #an agent is always available: exactly one request
    assert Resource_offset(m=0.041, p=1, n=3, rng=make_rng(1, "resource")) == 0.041
    
    #no agent can ever be available
    with pytest.raises(ValueError):
        Resource_offset(m=0.041, p=0, n=3, rng=make_rng(1, "resource"))
    
    #the mean waiting time is m divided by the probability of at least one available agent
    rng = make_rng(2, "resource")
    draws = [Resource_offset(m=0.041, p=0.5, n=3, rng=rng) for i in range(20000)]
    assert abs(np.mean(draws) - 0.041/0.875) < 0.02*0.041/0.875
    
    #without a generator, the global numpy stream is used
    np.random.seed(0)
    a = Resource_offset(m=0.041, p=0.5, n=3)
    np.random.seed(0)
    b = Resource_offset(m=0.041, p=0.5, n=3)
    assert a == b


def test_log_reproducible_for_all_processes():
    """
    Two event-logs from the same settings are identical in every column, for both process types and all entropy levels
    """
    import pandas as pd
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    for process_type in ["memoryless", "memory"]:
        for process_entropy in ["min_entropy", "med_entropy", "max_entropy"]:
            log1 = generate_eventlog(eventlog_settings(process_type=process_type, process_entropy=process_entropy, number_of_traces=60))
            log2 = generate_eventlog(eventlog_settings(process_type=process_type, process_entropy=process_entropy, number_of_traces=60))
            pd.testing.assert_frame_equal(log1, log2)

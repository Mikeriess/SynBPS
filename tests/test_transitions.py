import pytest

from test_eventlog import eventlog_settings


def states_reaching_end(P):
    """
    Reference implementation for the tests: the states (row indexes) of P from which END (the last state) is reachable
    """
    n = len(P)
    reaching = {n-1}
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i not in reaching and any(P[i, j] > 0 for j in reaching):
                reaching.add(i)
                changed = True
    return reaching


def test_states_not_reaching_absorption():
    """
    The helper finds the states of a closed class, and nothing else
    """
    import numpy as np
    from SynBPS.simulation.Memoryless_process.alg5_transition_matrix_med_entropy import States_not_reaching_absorption
    
    #a -> b or END, b -> c, c -> b, d -> c or END, END -> END: b and c form a closed class, d enters it or ends
    P = np.array([[0.0, 0.5, 0.0, 0.0, 0.5],
                  [0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.5, 0.0, 0.5],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])
    assert States_not_reaching_absorption(P) == [1, 2]
    
    #one transition out of the closed class is enough
    P[2, 4] = 0.5
    P[2, :] = P[2, :]/P[2, :].sum()
    assert States_not_reaching_absorption(P) == []


@pytest.mark.parametrize("statespace_size, n_transitions", [(5, 2), (10, 2), (5, 3), (10, 3)])
def test_med_entropy_absorption_reachable_from_every_state(statespace_size, n_transitions):
    """
    Every transition matrix of the memoryless med_entropy process lets every state reach the absorption state (issue 18),
    and the rows remain probability vectors
    """
    import numpy as np
    from SynBPS.simulation.simulation_helpers import make_D
    from SynBPS.simulation.Memoryless_process.alg5_transition_matrix_med_entropy import Generate_transition_matrix_med_ent
    
    D = make_D(statespace_size)
    
    for seed in range(300):
        P = Generate_transition_matrix_med_ent(D, n_tranitions=n_transitions, seed_value=seed).values
        
        assert len(states_reaching_end(P)) == len(P)
        assert np.allclose(P.sum(axis=1), 1)
        assert (P >= 0).all()
        
        #the absorption state only leads to itself
        assert P[-1, -1] == 1


def test_med_entropy_memoryless_closed_class_seed_terminates():
    """
    Seed 47 with statespace 5 and 2 transitions gave a closed class, from which the simulation never ended before issue 18
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    log = generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="med_entropy", 
                                              statespace_size=5, med_ent_n_transitions=2, 
                                              seed_value=47, number_of_traces=200))
    
    assert log.caseid.nunique() == 200


def test_max_len_memoryless():
    """
    The memoryless process raises when a trace exceeds max_len, as the process with memory does
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    with pytest.raises(Exception, match="max_len"):
        generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="max_entropy", 
                                            max_len=2, number_of_traces=100))
    
    #a large max_len does not interfere
    log = generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="max_entropy", 
                                              max_len=10000, number_of_traces=100))
    assert log.caseid.nunique() == 100


def test_kappa_default_and_effect():
    """
    med_ent_kappa (kappa in alg. 5) defaults to 2, is a floor for the sum of the absorption probabilities,
    and a higher value gives shorter traces
    """
    import numpy as np
    import pandas as pd
    from SynBPS.simulation.simulation_helpers import make_D
    from SynBPS.simulation.Memoryless_process.alg5_transition_matrix_med_entropy import Generate_transition_matrix_med_ent
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    D = make_D(5)
    
    #the sum of the last column (absorption, incl. the absorption state itself) is at least kappa
    for kappa in [1, 2, 3, 3.5]:
        for seed in range(50):
            P = Generate_transition_matrix_med_ent(D, n_tranitions=3, kappa=kappa, seed_value=seed).values
            assert P[:, -1].sum() >= kappa - 1e-9
            assert np.allclose(P.sum(axis=1), 1)
    
    #without the setting, the log is the one of kappa = 2
    settings = eventlog_settings(process_type="memoryless", process_entropy="med_entropy", number_of_traces=100)
    log_default = generate_eventlog(settings)
    log_2 = generate_eventlog(dict(settings, med_ent_kappa=2))
    pd.testing.assert_frame_equal(log_default, log_2)
    
    #values up to 1 add no transitions, as the absorption state contributes 1 by itself
    log_0 = generate_eventlog(dict(settings, med_ent_kappa=0))
    log_1 = generate_eventlog(dict(settings, med_ent_kappa=1))
    pd.testing.assert_frame_equal(log_0, log_1)
    
    #more absorption, shorter traces (on average over a few seeds)
    def mean_trace_length(kappa):
        lengths = []
        for seed in range(5):
            log = generate_eventlog(dict(settings, med_ent_kappa=kappa, seed_value=seed))
            lengths.append(len(log)/log.caseid.nunique())
        return np.mean(lengths)
    
    assert mean_trace_length(1) > mean_trace_length(2) > mean_trace_length(3.5)
    
    #the setting is ignored by the other processes, also when it is out of range for med_entropy
    generate_eventlog(eventlog_settings(process_type="memory", number_of_traces=10, med_ent_kappa=100))
    generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="max_entropy", number_of_traces=10, med_ent_kappa=100))


@pytest.mark.parametrize("kappa", [-1, 4, 100, "two"])
def test_kappa_invalid(kappa):
    """
    kappa must be between 0 and 1 + statespace_size/2 (3.5 for 5 activities), as the search becomes very slow above it
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    with pytest.raises(ValueError, match="med_ent_kappa"):
        generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="med_entropy", statespace_size=5, med_ent_kappa=kappa))


def test_resource_availability_mode():
    """
    resource_availability_mode: all_requests (default) counts every request, failed_requests only the failed ones.
    Both modes use the same random draws, so the logs differ by exactly m in h_t
    """
    import numpy as np
    import pandas as pd
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    from SynBPS.simulation.simulation_helpers import make_rng
    from SynBPS.simulation.Duration.duration_helpers import Resource_offset
    
    m, p, n = 0.041, 0.5, 3
    settings = eventlog_settings(process_type="memoryless", number_of_traces=300, resource_availability_m=m, resource_availability_p=p, resource_availability_n=n)
    
    #without the setting, the log is the one of all_requests
    log_default = generate_eventlog(settings)
    log_all = generate_eventlog(dict(settings, resource_availability_mode="all_requests"))
    pd.testing.assert_frame_equal(log_default, log_all)
    
    #failed_requests: h_t is exactly m lower for every event, everything else which does not depend on h_t is the same
    log_failed = generate_eventlog(dict(settings, resource_availability_mode="failed_requests"))
    assert np.allclose(log_all.h_t - log_failed.h_t, m)
    for column in ["activity", "z_t", "b_t", "v_t"]:
        assert log_all[column].tolist() == log_failed[column].tolist()
    
    #an agent is available at the first request with probability q = 1 - (1 - p)^n: then the offset is 0
    q = 1 - (1 - p)**n
    assert (log_all.h_t >= m).all()
    assert abs((log_failed.h_t == 0).mean() - q) < 0.05
    
    #means: m/q and m(1 - q)/q
    rng = make_rng(3, "resource")
    draws_all = np.array([Resource_offset(m=m, p=p, n=n, rng=rng, mode="all_requests") for i in range(20000)])
    rng = make_rng(3, "resource")
    draws_failed = np.array([Resource_offset(m=m, p=p, n=n, rng=rng, mode="failed_requests") for i in range(20000)])
    assert abs(draws_all.mean() - m/q) < 0.02*m/q
    assert abs(draws_failed.mean() - m*(1-q)/q) < 0.05*m/q
    
    #an agent which is always available: no offset in failed_requests, one request in all_requests
    log_p1 = generate_eventlog(dict(settings, resource_availability_p=1, resource_availability_mode="failed_requests"))
    assert (log_p1.h_t == 0).all()
    assert Resource_offset(m=m, p=1, n=n, rng=make_rng(1, "resource"), mode="all_requests") == m
    
    #invalid value
    with pytest.raises(ValueError, match="resource_availability_mode"):
        generate_eventlog(dict(settings, resource_availability_mode="none"))
    with pytest.raises(ValueError, match="resource_availability_mode"):
        Resource_offset(m=m, p=p, n=n, rng=make_rng(1, "resource"), mode="none")

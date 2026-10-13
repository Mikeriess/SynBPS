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
        np.random.seed(seed)
        P = Generate_transition_matrix_med_ent(D, n_tranitions=n_transitions).values
        
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

import pytest

from test_eventlog import eventlog_settings


def test_invalid_arguments_raise_valueerror(tmp_path):
    """
    Invalid arguments raise a ValueError instead of a string or a call to sys.exit()
    """
    from SynBPS.simulation.Memoryless_process.alg5_transition_matrix_med_entropy import Generate_transition_matrix_med_ent
    from SynBPS.simulation.alg6_memoryless_process_generator import Process_without_memory
    from SynBPS.simulation.Duration.alg9_trace_durations import Generate_time_variables
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    D = ["a","b","c","d","e"]
    
    #more transitions than states, or fewer than 2 transitions
    with pytest.raises(ValueError):
        Generate_transition_matrix_med_ent(["a","b"], n_tranitions=3)
    with pytest.raises(ValueError):
        Generate_transition_matrix_med_ent(D, n_tranitions=1)
    
    #unknown mode, and custom distributions without files
    with pytest.raises(ValueError):
        Process_without_memory(D, mode="high", num_traces=2)
    with pytest.raises(ValueError):
        Process_without_memory(D, mode="custom", num_traces=2)
    
    #custom Lambda with fewer rows than the longest trace
    lambda_file = tmp_path / "lambda.csv"
    lambda_file.write_text("a,b,c,d,e\n1,1,1,1,1\n")
    with pytest.raises(ValueError):
        Generate_time_variables(Theta=[["a","b","END"]], D=D, custom_distribution={"Lambda":str(lambda_file)})
    
    #custom distributions given, but process_entropy is not custom
    with pytest.raises(ValueError):
        generate_eventlog(eventlog_settings(process_type="memoryless", custom_distributions={"p0":"x","p":"x","Lambda":"x"}))
    
    #custom distributions with the process with memory
    with pytest.raises(ValueError):
        generate_eventlog(eventlog_settings(process_entropy="custom", custom_distributions={"p0":"x","p":"x","Lambda":"x"}))


def test_workweek():
    """
    Business hours: the none option has no closed hours, unknown values raise a ValueError
    """
    from SynBPS.simulation.simulation_helpers import make_workweek
    from SynBPS.simulation.Duration.duration_helpers import Deterministic_offset
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    #no closed hours
    assert make_workweek("none") == [[],[]]
    for q_t in [0, 0.25, 3.7, 6.99]:
        assert Deterministic_offset(make_workweek("none"), q_t) == 0
    
    #the presets are closed from the start of the week
    assert make_workweek("weekdays")[0][0] == 0.0
    assert make_workweek("all-week")[0][0] == 0.0
    
    #unknown value
    with pytest.raises(ValueError):
        make_workweek("weekend")
    
    #through generate_eventlog: no calendar offset with none
    log = generate_eventlog(eventlog_settings(Deterministic_offset_W="none"))
    assert (log.s_t == 0).all()
    assert log.start_hour.nunique() > 12
    
    #with weekdays, work starts in the open half of the day (continuous time 0.5 to 1 of each day, 06:00 to 18:00)
    log = generate_eventlog(eventlog_settings(Deterministic_offset_W="weekdays"))
    assert (log.s_t > 0).any()
    assert ((log.starttime % 1) >= 0.5 - 1e-9).all()

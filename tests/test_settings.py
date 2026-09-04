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

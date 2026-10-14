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


@pytest.mark.parametrize("overrides", [
    {"statespace_size":1},
    {"statespace_size":30},
    {"med_ent_n_transitions":1},
    {"process_type":"memoryless", "med_ent_n_transitions":6},
    {"process_memory":0},
    {"p_abs_min":1},
    {"p_abs_min":-0.1},
    {"max_len":1},
    {"resource_availability_p":0},
    {"resource_availability_p":1.5},
    {"resource_availability_n":0},
    {"resource_availability_m":-1},
    {"process_stability_scale":-1},
    {"inter_arrival_time":0},
    {"activity_duration_lambda_range":0},
    {"Deterministic_offset_u":0},
    {"Deterministic_offset_W":"weekend"},
    {"number_of_traces":0},
    {"seed_value":-1},
    {"process_type":"markov"},
    {"process_entropy":"high"},
    {"process_entropy":"custom"},
    {"process_entropy":"custom", "process_type":"memoryless"},
    {"process_entropy":"custom", "custom_distributions":{"p0":"x","p":"x","Lambda":"x"}},
    {"statespace_size":"five"},
])
def test_invalid_settings(overrides):
    """
    Every invalid setting raises a ValueError before the simulation starts
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    with pytest.raises(ValueError):
        generate_eventlog(eventlog_settings(**overrides))


def test_settings_message_lists_all_problems():
    """
    The error message names every invalid or missing setting at once
    """
    from SynBPS.simulation.simulate_eventlog import check_settings
    
    settings = eventlog_settings(statespace_size=1, number_of_traces=0)
    del settings["inter_arrival_time"]
    
    with pytest.raises(ValueError) as error:
        check_settings(settings)
    
    message = str(error.value)
    assert "statespace_size" in message
    assert "number_of_traces" in message
    assert "inter_arrival_time" in message


def test_valid_settings():
    """
    Settings which are valid must not raise: a design table row, and settings which are only checked for the process type or entropy in use
    """
    import pandas as pd
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    #a row of a design table is a pandas Series
    log = generate_eventlog(pd.Series(eventlog_settings(number_of_traces=20)))
    assert len(log.caseid.unique()) == 20
    
    #process_memory is not used by the memoryless process
    log = generate_eventlog(eventlog_settings(process_type="memoryless", process_memory=0, number_of_traces=20))
    assert len(log.caseid.unique()) == 20
    
    #med_ent_n_transitions is not used by min_entropy
    log = generate_eventlog(eventlog_settings(process_type="memoryless", process_entropy="min_entropy", med_ent_n_transitions=1, number_of_traces=20))
    assert len(log.caseid.unique()) == 20
    
    #the absorption state can be one of the transitions of the process with memory
    log = generate_eventlog(eventlog_settings(med_ent_n_transitions=6, number_of_traces=20))
    assert len(log.caseid.unique()) == 20
    
    #custom_distributions set to None counts as no custom distributions
    log = generate_eventlog(eventlog_settings(custom_distributions=None, number_of_traces=20))
    assert len(log.caseid.unique()) == 20


@pytest.mark.parametrize("datetime_offset", [0, 53, 54, 55, 56, 60])
def test_timestamps_weekdays_for_every_datetime_offset(datetime_offset):
    """
    The weekdays of the timestamps do not depend on datetime_offset: t = 0 is the first Sunday of the year at 18:00,
    so the business hours are Monday to Friday 06:00 to 18:00 for every offset
    """
    import pandas as pd
    from SynBPS.simulation.simulate_eventlog import generate_eventlog, timestamp_origin
    
    origin = timestamp_origin(datetime_offset)
    assert origin.year == 1970 + datetime_offset
    assert origin.day_name() == "Sunday"
    assert origin.day <= 7
    assert (origin.hour, origin.minute, origin.second) == (18, 0, 0)
    
    log = generate_eventlog(eventlog_settings(datetime_offset=datetime_offset, Deterministic_offset_W="weekdays", number_of_traces=200))
    
    #the arrival timestamp is the arrival time in continuous time, counted from the origin (rounded to whole seconds)
    arrival = origin + pd.to_timedelta((log.z_t * 24*60*60).round(), unit="s")
    pd.testing.assert_series_equal(log.arrival_datetime, arrival, check_names=False)
    
    #no activity starts in the weekend, and every activity starts between 06:00 and 18:00
    assert set(log.start_day.unique()) <= {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    assert log.start_day.nunique() == 5
    assert log.start_hour.between(6, 17).all()
    
    #an activity which waited for the opening hours starts at exactly 06:00:00
    waited = log.loc[log.s_t > 0, "start_datetime"]
    assert len(waited) > 0
    assert (waited.dt.strftime("%H:%M:%S") == "06:00:00").all()
    
    #the log starts in the requested year, within its first week
    assert log.arrival_datetime.min().year == 1970 + datetime_offset
    assert (log.arrival_datetime.min() - origin) < pd.Timedelta(days=7)


def test_activity_names():
    """
    The activities are the first letters of the alphabet, and the largest statespace uses all 29 letters
    """
    from SynBPS.simulation.simulation_helpers import make_D
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    assert make_D(5) == ["a", "b", "c", "d", "e"]
    assert make_D(26) == list("abcdefghijklmnopqrstuvwxyz")
    assert make_D(29) == list("abcdefghijklmnopqrstuvwxyz") + ["æ", "ø", "å"]
    assert len(set(make_D(29))) == 29
    
    with pytest.raises(ValueError):
        make_D(30)
    
    #the activities of a log are the letters of its statespace
    log = generate_eventlog(eventlog_settings(statespace_size=5, process_entropy="max_entropy", number_of_traces=50))
    assert set(log.activity.unique()) <= set(make_D(5))
    assert "a" in log.activity.unique()
    
    #the largest statespace works end to end
    log = generate_eventlog(eventlog_settings(statespace_size=29, process_type="memoryless", process_entropy="max_entropy", number_of_traces=50))
    assert set(log.activity.unique()) <= set(make_D(29))


def test_max_entropy_with_memory_warns():
    """
    max_entropy is the uniform extreme, where process_memory has no effect: check_settings warns when the
    order is above 1 (issue 48), and not for the other levels, for order 1, or for the memoryless process
    """
    import warnings
    from SynBPS.simulation.simulate_eventlog import check_settings
    
    with pytest.warns(UserWarning, match="process_memory has no effect with max_entropy"):
        check_settings(eventlog_settings(process_type="memory", process_entropy="max_entropy", process_memory=2))
    
    for overrides in [dict(process_type="memory", process_entropy="max_entropy", process_memory=1),
                      dict(process_type="memory", process_entropy="med_entropy", process_memory=2),
                      dict(process_type="memory", process_entropy="min_entropy", process_memory=2),
                      dict(process_type="memoryless", process_entropy="max_entropy", process_memory=2)]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_settings(eventlog_settings(**overrides))

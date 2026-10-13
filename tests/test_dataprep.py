import pytest

from test_eventlog import eventlog_settings


@pytest.fixture(scope="module")
def log():
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    return generate_eventlog(eventlog_settings(process_type="memoryless", number_of_traces=60))


@pytest.fixture(scope="module")
def data(log):
    from SynBPS.dataprep.prepare import prefix_data
    return prefix_data(log)


def test_prefix_data_shapes(log, data):
    """
    One observation per event except the last of every trace with more than one event; arrays and inference tables agree
    """
    import numpy as np
    
    kept = log.groupby("caseid").size()
    kept = kept[kept > 1]
    observations = int((kept - 1).sum())
    prefix_length = int(kept.max())
    activities = log.activity.nunique()
    
    n_train = data["x_train"].shape[0]
    n_test = data["x_test"].shape[0]
    assert n_train + n_test == observations
    assert n_train > n_test > 0
    
    # 6 time features plus the one-hot activities
    assert data["x_train"].shape == (n_train, prefix_length, 6 + activities)
    assert data["x_test"].shape == (n_test, prefix_length, 6 + activities)
    assert data["y_train"].shape == (n_train, 1)
    assert data["y_t_train"].shape == (n_train, 1)
    assert data["y_a_train"].shape == (n_train, activities)
    assert len(data["Inference_train"]) == n_train
    assert len(data["Inference_test"]) == n_test
    assert data["scaling"]["mode"] is None
    
    #the next activity is one-hot
    assert (data["y_a_train"].sum(axis=1) == 1).all()


def test_prefix_data_event_number_and_links(log, data):
    """
    event_number and prefix_number count the observed events from 1, prefix_date is the start of the last observed event,
    and log_caseid links the observation to the event-log
    """
    import numpy as np
    import pandas as pd
    
    inference = pd.concat([data["Inference_train"], data["Inference_test"]])
    x = np.concatenate([data["x_train"], data["x_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])
    
    assert inference.prefix_number.min() == 1
    assert (inference.prefix_number == inference.event_number).all()
    assert (inference.prefixes == inference.num_events - 1).all()
    assert (inference.prefix_number <= inference.prefixes).all()
    assert (inference.SEQID == inference.caseid.astype(str) + "_" + inference.prefix_number.astype(str)).all()
    
    for i, row in enumerate(inference.itertuples()):
        trace = log.loc[log.caseid == str(row.log_caseid)].sort_values("activity_no")
        assert len(trace) == row.num_events
        
        #the last observed event
        last = trace.iloc[row.prefix_number - 1]
        assert row.prefix_date == last.start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        #its row in the sequence is the last row, and event_number is its number
        assert x[i, -1, 0] == row.prefix_number
        assert x[i, -1, 1] == last.start_datetime.hour*3600 + last.start_datetime.minute*60 + last.start_datetime.second
        assert x[i, -1, 2] == last.start_datetime.weekday() + 1
        assert x[i, -1, 3] == last.start_datetime.hour
        
        #the rows before the observed events are padding
        assert (x[i, :-row.prefix_number, :] == 0).all()
        
        #the remaining time is the time to the start of the last event of the trace
        assert abs(y[i, 0] - (trace.iloc[-1].start_datetime - last.start_datetime).total_seconds()) < 1
    
    #every trace with more than one event is in the tables, with its caseid of the log
    sizes = log.groupby("caseid").size()
    assert set(inference.log_caseid.astype(str)) == set(sizes[sizes > 1].index)


def test_prefix_data_extra_features(log):
    """
    Columns of the log can be added as one-hot or numeric features
    """
    from SynBPS.dataprep.prepare import prefix_data
    
    data = prefix_data(log, category_cols=["start_day"], numeric_cols=["h_t", "s_t"])
    
    weekdays = log.start_day.nunique()
    activities = log.activity.nunique()
    assert data["x_train"].shape[2] == 6 + activities + weekdays + 2
    
    #with one-hot time features, dayofweek and hourofday are replaced by their dummies
    data = prefix_data(log, dummify_time_features=True)
    assert data["x_train"].shape[2] == 4 + activities + log.start_datetime.dt.weekday.nunique() + log.start_datetime.dt.hour.nunique()
    
    with pytest.raises(ValueError, match="category_cols"):
        prefix_data(log, category_cols=["weekday"])
    with pytest.raises(ValueError, match="window_position"):
        prefix_data(log, window_position="middle")
    with pytest.raises(ValueError, match="scaling"):
        prefix_data(log, scaling="minmax")


@pytest.mark.parametrize("mode", ["standardize", "normalize"])
def test_prefix_data_scaling(log, data, mode):
    """
    The numeric features are scaled with the parameters of the train set, one-hot features and padding are untouched, targets are not scaled
    """
    import numpy as np
    from SynBPS.dataprep.prepare import prefix_data
    
    scaled = prefix_data(log, numeric_cols=["h_t"], scaling=mode)
    plain = prefix_data(log, numeric_cols=["h_t"])
    
    assert scaled["scaling"]["mode"] == mode
    assert set(scaled["scaling"]["columns"]) == {"event_number", "timesincemidnight", "dayofweek", "hourofday", "timesincestart", "timesincelastev", "num_h_t"}
    
    x_scaled, x_plain = scaled["x_train"], plain["x_train"]
    activities = log.activity.nunique()
    
    #the one-hot columns and the padding are the same
    assert np.array_equal(x_scaled[:, :, 6:6 + activities], x_plain[:, :, 6:6 + activities])
    padding = (x_plain == 0).all(axis=2)
    assert (x_scaled[padding] == 0).all()
    
    #the observed rows are scaled with the parameters of the train set
    observed = ~padding
    for j, column in enumerate(scaled["scaling"]["columns"]):
        index = j if j < 6 else 6 + activities
        values = x_plain[:, :, index][observed].astype(float)
        parameters = scaled["scaling"]["parameters"][column]
        if mode == "standardize":
            assert abs(parameters["mean"] - values.mean()) < 1e-6
            expected = (values - parameters["mean"]) / (parameters["std"] if parameters["std"] > 0 else 1.0)
        if mode == "normalize":
            assert parameters["min"] == values.min()
            expected = (values - parameters["min"]) / ((parameters["max"] - parameters["min"]) if parameters["max"] > parameters["min"] else 1.0)
        assert np.allclose(x_scaled[:, :, index][observed].astype(float), expected)
    
    #the test set is scaled with the same parameters, and the targets are not scaled
    assert not np.array_equal(scaled["x_test"][:, -1, 1], plain["x_test"][:, -1, 1])
    assert np.array_equal(scaled["y_train"], plain["y_train"])
    assert np.array_equal(scaled["y_t_test"], plain["y_t_test"])
    assert scaled["Inference_test"].y.tolist() == plain["Inference_test"].y.tolist()

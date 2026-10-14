import pytest
import warnings

from test_eventlog import eventlog_settings


@pytest.fixture(scope="module")
def log():
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    return generate_eventlog(eventlog_settings(process_type="memoryless", number_of_traces=120, seed_value=5))


def case_table(log):
    """
    The number of events and the start of the first and last event of every case, from the log
    """
    import pandas as pd
    t = log.sort_values(["caseid", "activity_no"]).groupby("caseid", sort=False)
    return pd.DataFrame({"num_events": t.size(), "start": t.start_datetime.first(), "end": t.start_datetime.last()})


def test_split_cases_rules(log):
    """
    One row per case, single-event cases dropped, the split date between the train and the test cases,
    a share of trainsize of the kept cases in the train set, spanning cases in the train set
    """
    import pandas as pd
    from SynBPS.dataprep.prepare import split_cases
    
    cases = split_cases(log, trainsize=0.7)
    table = case_table(log)
    
    #one row per case of the log, in the order of the log, with the columns of the docs
    assert cases.log_caseid.tolist() == log.caseid.unique().tolist()
    assert cases.columns.tolist() == ["log_caseid", "num_events", "start", "end", "set", "dropped_reason", "split_date"]
    assert (cases.set_index("log_caseid").num_events == table.num_events.reindex(cases.log_caseid)).all()
    
    #rule 1: single-event cases are dropped, nothing else is
    assert (cases.loc[cases.num_events == 1, "set"] == "dropped").all()
    assert (cases.loc[cases.num_events == 1, "dropped_reason"] == "single event").all()
    assert (cases.loc[cases.num_events > 1, "set"] != "dropped").all()
    assert (cases.loc[cases.num_events > 1, "dropped_reason"] == "").all()
    
    #rule 2 and 3: a share of trainsize of the kept cases starts before the split date
    kept = cases.loc[cases.set != "dropped"]
    split_date = cases.split_date.iloc[0]
    assert cases.split_date.nunique() == 1
    assert (kept.loc[kept.set == "train", "start"] < split_date).all()
    assert (kept.loc[kept.set == "test", "start"] >= split_date).all()
    #as close to trainsize as the cases allow: cases which start at the same time cannot be separated by a date
    starts = kept.start.sort_values().values
    achievable = {int((starts < d).sum()) for d in set(starts)}
    target = round(0.7 * len(kept))
    assert abs((kept.set == "train").sum() - target) == min(abs(a - target) for a in achievable)
    
    #the split date is the start of the first test case
    assert split_date == kept.loc[kept.set == "test", "start"].min()
    
    #rule 4: spanning cases stay in the train set by default, and are dropped on request
    spanning = (kept.start < split_date) & (kept.end >= split_date)
    assert spanning.sum() > 0
    assert (kept.loc[spanning, "set"] == "train").all()
    
    dropped = split_cases(log, trainsize=0.7, spanning_cases="drop")
    assert dropped.split_date.iloc[0] == split_date
    assert (dropped.loc[dropped.log_caseid.isin(kept.loc[spanning, "log_caseid"]), "set"] == "dropped").all()
    assert (dropped.loc[dropped.log_caseid.isin(kept.loc[spanning, "log_caseid"]), "dropped_reason"] == "spans the split date").all()
    assert (dropped.loc[dropped.log_caseid.isin(kept.loc[~spanning, "log_caseid"]), "set"] == kept.loc[~spanning, "set"].values).all()


def test_split_cases_by_date_matches_previous_rule(log):
    """
    split_by date gives the rule of MakeSplitCriterion: the earliest event plus trainsize of the days of the log
    """
    import pandas as pd
    from SynBPS.dataprep.prepare import split_cases
    from SynBPS.dataprep.dataprep_helperfunctions import MakeSplitCriterion
    
    cases = split_cases(log, trainsize=0.7, split_by="date")
    
    #the previous rule, on the cases with more than one event
    sizes = log.groupby("caseid").size()
    df = log.loc[log.caseid.isin(sizes[sizes > 1].index), ["caseid", "start_datetime"]].rename(columns={"caseid":"id", "start_datetime":"time"})
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    previous = MakeSplitCriterion(df.copy(), trainsize=0.7, mode="event")
    
    merged = pd.merge(cases, previous, left_on="log_caseid", right_on="caseid")
    assert len(merged) == (cases.set != "dropped").sum()
    assert ((merged.set == "train") == merged.trainset).all()
    
    #the share of cases is not trainsize with this rule, in general
    kept = cases.loc[cases.set != "dropped"]
    assert abs((kept.set == "train").mean() - 0.7) > 0.01


def test_split_cases_explicit_date_and_single_events(log):
    """
    An explicit split date replaces trainsize, and single-event cases can be kept
    """
    import pandas as pd
    from SynBPS.dataprep.prepare import split_cases
    
    table = case_table(log)
    date = table.start.sort_values().iloc[len(table)//2]
    
    cases = split_cases(log, split_date=str(date), trainsize=5)
    assert cases.split_date.iloc[0] == pd.Timestamp(date)
    kept = cases.loc[cases.set != "dropped"]
    assert ((kept.start < date) == (kept.set == "train")).all()
    
    cases = split_cases(log, drop_single_event_cases=False)
    assert (cases.set != "dropped").all()
    assert abs((cases.set == "train").sum() - round(0.7 * len(cases))) <= (cases.start == cases.split_date.iloc[0]).sum()


@pytest.mark.parametrize("kwargs", [{"split_by":"time"}, {"spanning_cases":"test"}, {"trainsize":0}, {"trainsize":1}, {"split_date":"not a date"}])
def test_split_cases_invalid(log, kwargs):
    from SynBPS.dataprep.prepare import split_cases
    
    with pytest.raises(ValueError):
        split_cases(log, **kwargs)
    
    with pytest.raises(ValueError, match="generate_eventlog"):
        split_cases(log.drop(columns="activity_no"))


def test_prefix_data_uses_split_cases(log):
    """
    The assignment of prefix_data equals split_cases, is returned in cases, and can be given back as case_sets
    """
    import numpy as np
    import pandas as pd
    from SynBPS.dataprep.prepare import split_cases, prefix_data
    
    for kwargs in [{}, {"spanning_cases":"drop", "trainsize":0.5}, {"split_by":"date"}]:
        cases = split_cases(log, **kwargs)
        data = prefix_data(log, **kwargs)
        
        pd.testing.assert_frame_equal(data["cases"], cases)
        assert data["split_date"] == cases.split_date.iloc[0]
        
        #the inference tables hold exactly the train and the test cases
        assert set(data["Inference_train"].log_caseid.astype(str)) == set(cases.loc[cases.set == "train", "log_caseid"].astype(str))
        assert set(data["Inference_test"].log_caseid.astype(str)) == set(cases.loc[cases.set == "test", "log_caseid"].astype(str))
        
        #the same split given as case_sets gives the same arrays
        same = prefix_data(log, case_sets=cases)
        for key in ["x_train", "x_test", "y_train", "y_test", "y_a_train", "y_a_test", "y_t_train", "y_t_test"]:
            assert np.array_equal(data[key].astype(float), same[key].astype(float))
        pd.testing.assert_frame_equal(data["Inference_test"], same["Inference_test"])
    
    #case_sets must cover the log
    with pytest.raises(ValueError, match="case_sets"):
        prefix_data(log, case_sets=cases.iloc[:10])
    with pytest.raises(ValueError, match="case_sets"):
        prefix_data(log, case_sets=cases.drop(columns="set"))


def test_prefix_data_include_end(log):
    """
    With include_end the last event of every trace is an observation with END as the next activity
    and 0 as remaining time, and single-event cases can be kept
    """
    import numpy as np
    import pandas as pd
    from SynBPS.dataprep.prepare import prefix_data
    
    plain = prefix_data(log)
    with_end = prefix_data(log, include_end=True)
    
    cases = with_end["cases"]
    kept = cases.loc[cases.set != "dropped"]
    
    #one observation per event, instead of one per event but the last
    assert with_end["x_train"].shape[0] + with_end["x_test"].shape[0] == kept.num_events.sum()
    assert plain["x_train"].shape[0] + plain["x_test"].shape[0] == (kept.num_events - 1).sum()
    
    #END is a column of y_a, the first one
    activities = log.activity.nunique()
    assert with_end["y_a_train"].shape[1] == activities + 1
    assert plain["y_a_train"].shape[1] == activities
    assert with_end["Inference_train"].columns[with_end["Inference_train"].columns.str.startswith("y_a_t1_")][0] == "y_a_t1_1"
    
    #the last observation of every case: END, remaining time 0, time to next event 0
    inference = pd.concat([with_end["Inference_train"], with_end["Inference_test"]])
    last = inference.loc[inference.prefix_number == inference.num_events]
    assert len(last) == len(kept)
    assert (last.prefixes == last.num_events).all()
    assert (last.y == 0).all()
    assert (last.y_t == 0).all()
    assert last.y_a_t1_1.all()
    assert not inference.loc[inference.prefix_number < inference.num_events, "y_a_t1_1"].any()
    
    #the other observations are the ones of the default
    plain_inference = pd.concat([plain["Inference_train"], plain["Inference_test"]])
    assert set(plain_inference.SEQID) == set(inference.loc[inference.prefix_number < inference.num_events, "SEQID"])
    
    #single-event cases are observations with include_end
    with_single = prefix_data(log, include_end=True, drop_single_event_cases=False)
    singles = with_single["cases"].loc[with_single["cases"].num_events == 1]
    assert len(singles) > 0
    assert (singles.set != "dropped").all()
    inference = pd.concat([with_single["Inference_train"], with_single["Inference_test"]])
    assert set(singles.log_caseid.astype(str)) <= set(inference.log_caseid.astype(str))
    
    #without include_end they give no observation, which is warned about
    with pytest.warns(UserWarning, match="include_end"):
        without = prefix_data(log, drop_single_event_cases=False)
    inference = pd.concat([without["Inference_train"], without["Inference_test"]])
    assert not set(singles.log_caseid.astype(str)) & set(inference.log_caseid.astype(str))

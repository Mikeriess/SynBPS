import pytest

from test_eventlog import eventlog_settings


def custom_files(path, D=["a","b","c"]):
    """
    Custom distribution files for a small memoryless process: p0, p and Lambda
    """
    import pandas as pd
    
    pd.DataFrame({"p0":[0.5, 0.3, 0.2]}).to_csv(path / "p0.csv", index=False)
    P = [[0.2, 0.5, 0.0, 0.3],
         [0.0, 0.2, 0.5, 0.3],
         [0.4, 0.0, 0.2, 0.4],
         [0.0, 0.0, 0.0, 1.0]]
    pd.DataFrame(P, columns=D + ["END"]).to_csv(path / "p.csv", index=False)
    pd.DataFrame([[1.0, 2.0, 3.0]]*30, columns=D).to_csv(path / "lambda.csv", index=False)
    
    return {"p0":str(path / "p0.csv"), "p":str(path / "p.csv"), "Lambda":str(path / "lambda.csv")}


def all_settings(tmp_path):
    """
    Settings for both process types, every entropy level and custom distributions
    """
    settings = []
    for process_type in ["memoryless", "memory"]:
        for process_entropy in ["min_entropy", "med_entropy", "max_entropy"]:
            settings.append(eventlog_settings(process_type=process_type, process_entropy=process_entropy, number_of_traces=40))
    settings.append(eventlog_settings(process_type="memoryless", process_entropy="custom", statespace_size=3, number_of_traces=40,
                                      custom_distributions=custom_files(tmp_path)))
    return settings


def assert_same_dgp(dgp1, dgp2):
    import numpy as np
    
    assert dgp1.keys() == dgp2.keys()
    for key in ["process_type", "mode", "D", "D_abs", "K", "time_settings", "max_len", "seed_value", "datetime_offset"]:
        assert dgp1[key] == dgp2[key]
    assert np.array_equal(dgp1["P0"], dgp2["P0"])
    assert dgp1["Phi"].keys() == dgp2["Phi"].keys()
    for k in dgp1["Phi"]:
        assert dgp1["Phi"][k].keys() == dgp2["Phi"][k].keys()
        for context in dgp1["Phi"][k]:
            assert np.array_equal(dgp1["Phi"][k][context], dgp2["Phi"][k][context])
    assert (dgp1["P"] is None) == (dgp2["P"] is None)
    if dgp1["P"] is not None:
        assert dgp1["P"].equals(dgp2["P"])
    assert dgp1["Lambda"].equals(dgp2["Lambda"])


def replay_control_flow(dgp, num_traces):
    """
    Samples the traces again from P0 and Phi with the sampling stream of the seed, as the generators do
    """
    from SynBPS.simulation.simulation_helpers import make_rng
    
    stream = "homc_sampling" if dgp["process_type"] == "memory" else "memoryless_sampling"
    rng = make_rng(dgp["seed_value"], stream)
    D, D_abs, K, P0, Phi = dgp["D"], dgp["D_abs"], dgp["K"], dgp["P0"], dgp["Phi"]
    
    traces = []
    for i in range(num_traces):
        trace = [D[rng.choice(len(D), p=P0)]]
        while trace[-1] != "END":
            k = min(len(trace), K)
            trace.append(D_abs[rng.choice(len(D_abs), p=Phi[k][tuple(trace[-k:])])])
        traces.append(trace[:-1])
    return traces


def test_recover_dgp_equals_the_process_of_the_log(tmp_path):
    """
    generate_eventlog with return_dgp returns the log and its process, recover_dgp returns the same
    process without the log, and the log is the same with and without return_dgp (issue 49)
    """
    import pandas as pd
    from SynBPS.simulation.simulate_eventlog import generate_eventlog, recover_dgp
    
    for settings in all_settings(tmp_path):
        log = generate_eventlog(settings)
        log2, dgp = generate_eventlog(settings, return_dgp=True)
        pd.testing.assert_frame_equal(log, log2)
        
        assert_same_dgp(dgp, recover_dgp(settings))
        
        assert dgp["process_type"] == settings["process_type"]
        assert dgp["mode"] == settings["process_entropy"]
        assert dgp["D_abs"] == dgp["D"] + ["END"]
        assert dgp["K"] == (settings["process_memory"] if settings["process_type"] == "memory" else 1)
        assert dgp["time_settings"]["resource_availability_p"] == settings["resource_availability_p"]


def test_phi_has_one_structure_for_both_process_types(tmp_path):
    """
    Phi[k][context] is a probability vector over D_abs for k = 1 to K, for both process types.
    For the memoryless process Phi[1] holds the rows of P, and P0 is an array over D
    """
    import numpy as np
    from SynBPS.simulation.simulate_eventlog import recover_dgp
    
    for settings in all_settings(tmp_path):
        dgp = recover_dgp(settings)
        D, D_abs = dgp["D"], dgp["D_abs"]
        
        assert isinstance(dgp["P0"], np.ndarray) and dgp["P0"].shape == (len(D),)
        assert np.isclose(dgp["P0"].sum(), 1)
        
        assert sorted(dgp["Phi"].keys()) == list(range(1, dgp["K"]+1))
        for k, table in dgp["Phi"].items():
            assert len(table) == len(D)**k
            for context, row in table.items():
                assert len(context) == k and all(d in D for d in context)
                assert row.shape == (len(D_abs),) and np.isclose(row.sum(), 1) and (row >= 0).all()
        
        if dgp["process_type"] == "memoryless":
            assert list(dgp["P"].index) == D_abs and list(dgp["P"].columns) == D_abs
            for d in D:
                assert np.array_equal(dgp["Phi"][1][(d,)], dgp["P"].loc[d].values)
        else:
            assert dgp["P"] is None


def test_phi_reproduces_the_control_flow_of_the_log(tmp_path):
    """
    The traces of the log are the traces sampled from P0 and Phi with the sampling stream of the seed,
    and every transition of the log has a positive probability under Phi
    """
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    
    for settings in all_settings(tmp_path):
        log, dgp = generate_eventlog(settings, return_dgp=True)
        
        traces = [trace.sort_values("activity_no").activity.tolist() for _, trace in log.groupby("caseid", sort=False)]
        assert replay_control_flow(dgp, len(traces)) == traces
        
        for trace in traces:
            assert dgp["P0"][dgp["D"].index(trace[0])] > 0
            activities = trace + ["END"]
            for t in range(1, len(activities)):
                k = min(t, dgp["K"])
                assert dgp["Phi"][k][tuple(activities[t-k:t])][dgp["D_abs"].index(activities[t])] > 0


def test_lambda_has_a_row_for_every_position(tmp_path):
    """
    Lambda has max_len rows, so the mean duration of every activity at every possible position is known,
    and the durations of the log are drawn from it
    """
    import numpy as np
    from SynBPS.simulation.simulate_eventlog import generate_eventlog, recover_dgp
    
    for max_len in [500, 10000]:
        settings = eventlog_settings(process_type="memory", number_of_traces=40, max_len=max_len)
        dgp = recover_dgp(settings)
        assert dgp["Lambda"].shape == (max_len, len(dgp["D"]))
        assert list(dgp["Lambda"].columns) == dgp["D"]
        assert (dgp["Lambda"] > 0).all().all()
    
    #the first rows do not depend on max_len
    small = recover_dgp(eventlog_settings(process_type="memory", max_len=500))["Lambda"]
    big = recover_dgp(eventlog_settings(process_type="memory", max_len=10000))["Lambda"]
    assert small.equals(big.iloc[:500])
    
    #custom distributions: Lambda is the file
    settings = eventlog_settings(process_type="memoryless", process_entropy="custom", statespace_size=3, number_of_traces=20,
                                 custom_distributions=custom_files(tmp_path))
    dgp = recover_dgp(settings)
    assert dgp["Lambda"].shape == (30, 3)
    assert dgp["P"].loc["a", "b"] == 0.5
    assert np.array_equal(dgp["P0"], [0.5, 0.3, 0.2])

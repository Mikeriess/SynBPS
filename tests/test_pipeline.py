import pytest

from test_eventlog import eventlog_settings


def design_table(path, seeds):
    """
    A design table with one run per seed, for a small memoryless process
    """
    from SynBPS.design.DoE import make_design_table
    
    base = eventlog_settings(process_type="memoryless", number_of_traces=20)
    factors = {key: [value] for key, value in base.items()}
    factors["seed_value"] = seeds
    
    df = make_design_table(factors)
    df.to_csv(path / "design_table.csv", index=False)
    
    return df


def dataprep(log):
    return log


def training(data):
    return data[["caseid"]].drop_duplicates()


def evaluation(inference):
    return {"metric": len(inference)}


def test_run_experiments_keeps_results_when_resumed(tmp_path):
    """
    The results of earlier calls are kept when the experiment is continued (issue 19)
    """
    import pandas as pd
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    out = str(tmp_path) + "/"
    df = design_table(tmp_path, seeds=[1, 2, 3])
    
    #first session: run 3 is marked as done beforehand, so runs 1 and 2 are performed
    df.loc[2, "Done"] = 1
    df.to_csv(tmp_path / "design_table.csv", index=False)
    
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert results.RUN.tolist() == [1, 2]
    assert results.metric.tolist() == [20, 20]
    assert pd.read_csv(tmp_path / "results.csv").RUN.tolist() == [1, 2]
    
    #second session: run 3 is not done after all
    df = pd.read_csv(tmp_path / "design_table.csv")
    assert df.Done.tolist() == [1, 1, 1]
    df.loc[2, "Done"] = 0
    df.to_csv(tmp_path / "design_table.csv", index=False)
    
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert results.RUN.tolist() == [1, 2, 3]
    assert pd.read_csv(tmp_path / "results.csv").RUN.tolist() == [1, 2, 3]
    
    #third session: nothing left to do, the results so far are returned
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert results.RUN.tolist() == [1, 2, 3]
    assert pd.read_csv(tmp_path / "results.csv").RUN.tolist() == [1, 2, 3]
    
    #a run performed again replaces its earlier row
    df = pd.read_csv(tmp_path / "design_table.csv")
    df.loc[0, "Done"] = 0
    df.to_csv(tmp_path / "design_table.csv", index=False)
    
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert sorted(results.RUN.tolist()) == [1, 2, 3]
    assert len(pd.read_csv(tmp_path / "results.csv")) == 3


def test_run_experiments_marks_failed_runs(tmp_path, capsys):
    """
    A run which raises is marked with Failure = 1, the error is printed, and the other runs are performed
    """
    import pandas as pd
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    out = str(tmp_path) + "/"
    design_table(tmp_path, seeds=[1, 2, 3])
    
    def training_fails_on_seed_2(data):
        if data.caseid.nunique() == 20 and data.v_t.sum() == training_fails_on_seed_2.v_sum:
            raise RuntimeError("model did not converge")
        return training(data)
    
    #the sum of the durations identifies the log of seed 2
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    training_fails_on_seed_2.v_sum = generate_eventlog(eventlog_settings(process_type="memoryless", number_of_traces=20, seed_value=2)).v_t.sum()
    
    results = run_experiments(dataprep, training_fails_on_seed_2, evaluation, output_dir=out)
    
    assert results.RUN.tolist() == [1, 3]
    
    df = pd.read_csv(tmp_path / "design_table.csv")
    assert df.Done.tolist() == [1, 0, 1]
    assert df.Failure.tolist() == [0, 1, 0]
    
    printed = capsys.readouterr().out
    assert "Run 1 failed" in printed
    assert "model did not converge" in printed
    
    #the failed run is retried when the function is called again, and succeeds with a working training function
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert sorted(results.RUN.tolist()) == [1, 2, 3]
    
    df = pd.read_csv(tmp_path / "design_table.csv")
    assert df.Done.tolist() == [1, 1, 1]
    assert df.Failure.tolist() == [0, 0, 0]


def test_run_experiments_without_failure_column(tmp_path):
    """
    A design table without the Failure column (not created by make_design_table) works as well
    """
    import pandas as pd
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    out = str(tmp_path) + "/"
    df = design_table(tmp_path, seeds=[1])
    df.drop(columns="Failure").to_csv(tmp_path / "design_table.csv", index=False)
    
    results = run_experiments(dataprep, training, evaluation, output_dir=out)
    assert results.RUN.tolist() == [1]
    assert pd.read_csv(tmp_path / "design_table.csv").Failure.tolist() == [0]


def test_design_table_without_statespace_size(tmp_path):
    """
    A design where statespace_size is not a factor can be created and run,
    with the remaining settings added as constant columns (issue 41)
    """
    import pandas as pd
    from SynBPS.design.DoE import make_design_table
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    df = make_design_table({"Deterministic_offset_W": ["none", "weekdays"], "seed_value": [1, 2]})
    assert len(df) == 4
    assert df.RUN.tolist() == [1, 2, 3, 4]
    assert "statespace_size" not in df.columns
    
    #the settings which are not factors are added as constant columns
    base = eventlog_settings(process_type="memoryless", number_of_traces=20)
    for key, value in base.items():
        if key not in df.columns:
            df[key] = value
    df.to_csv(tmp_path / "design_table.csv", index=False)
    
    results = run_experiments(dataprep, training, evaluation, output_dir=str(tmp_path) + "/")
    assert results.RUN.tolist() == [1, 2, 3, 4]
    assert results.metric.tolist() == [20, 20, 20, 20]


def test_user_functions_receive_the_settings(tmp_path):
    """
    A user function which declares a settings argument (or **kwargs) receives the row of the design
    table, and a function without it is called as before (issue 42)
    """
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    out = str(tmp_path) + "/"
    design_table(tmp_path, seeds=[11, 12])
    received = []
    
    def dataprep_with_settings(log, settings=None):
        received.append(("dataprep", settings))
        return log
    
    def training_with_kwargs(data, **kwargs):
        received.append(("training", kwargs["settings"]))
        return training(data)
    
    results = run_experiments(dataprep_with_settings, training_with_kwargs, evaluation, output_dir=out)
    assert results.RUN.tolist() == [1, 2]
    
    #both functions received the settings of both runs, in order
    assert [name for name, _ in received] == ["dataprep", "training", "dataprep", "training"]
    assert [settings["seed_value"] for _, settings in received] == [11, 11, 12, 12]
    assert [settings["idx"] for _, settings in received] == [0, 0, 1, 1]
    
    #the settings are the row of the design table, without the characteristics of the event-log
    settings = received[0][1]
    assert isinstance(settings, dict)
    assert settings["process_type"] == "memoryless"
    assert settings["RUN"] == 1
    assert "num_events" not in settings


def test_training_function_recovers_the_process_from_the_settings(tmp_path):
    """
    The settings are what a training function needs to recover the data generating process:
    the transition tables of the process with memory, recovered with create_homc, give a positive
    probability to every transition of the event-log (issue 42)
    """
    from SynBPS.design.DoE import make_design_table
    from SynBPS.simulation.simulation_pipeline import run_experiments
    
    base = eventlog_settings(process_type="memory", process_memory=2, number_of_traces=30)
    factors = {key: [value] for key, value in base.items()}
    factors["seed_value"] = [21, 22]
    make_design_table(factors).to_csv(tmp_path / "design_table.csv", index=False)
    
    def training_from_process(log, settings=None):
        from SynBPS.simulation.simulation_helpers import make_D
        from SynBPS.simulation.alg7_memory_process_generator import create_homc
        
        homc = create_homc(D=make_D(int(settings["statespace_size"])),
                           K=int(settings["process_memory"]),
                           mode=settings["process_entropy"],
                           n_transitions=int(settings["med_ent_n_transitions"]),
                           p_abs_min=float(settings["p_abs_min"]),
                           seed_value=int(settings["seed_value"]))
        
        #the probability of every observed transition under the recovered process
        probabilities = []
        for _, trace in log.groupby("caseid", sort=False):
            activities = trace.activity.tolist() + ["END"]
            for t in range(1, len(activities)):
                i = min(t, homc["K"])
                context = tuple(activities[t-i:t])
                probabilities.append(homc["Phi"][i][context][homc["D_abs"].index(activities[t])])
        
        import pandas as pd
        return pd.DataFrame({"p": probabilities})
    
    def evaluation_min_p(inference):
        return {"min_p": inference.p.min(), "transitions": len(inference)}
    
    results = run_experiments(dataprep, training_from_process, evaluation_min_p, output_dir=str(tmp_path) + "/")
    assert results.RUN.tolist() == [1, 2]
    assert (results.min_p > 0).all()
    assert (results.transitions == results.num_events).all()

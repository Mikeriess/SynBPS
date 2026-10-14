#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def run_experiments(dataprep_function, training_function, eval_function, store_eventlogs=False, output_dir="data/", out_file="results.csv", design_table="design_table.csv", verbose=False):
    """
    Function for running experiments

    Every run of the design table which is not marked as done (Done == 0) is performed: the event-log is generated,
    prepared with dataprep_function, used for training with training_function and evaluated with eval_function.
    The settings, the characteristics of the event-log and the evaluation metrics of the run are appended to the
    results file, and the run is marked as done in the design table. Both files are written after every run, so an
    interrupted experiment can be continued by calling the function again: the results of the earlier runs are kept.

    A run which raises an exception is marked with Failure = 1 in the design table (Done stays 0), the error is
    printed, and the experiment continues with the next run. Calling the function again retries the failed runs.

    Each of the three functions can declare an optional keyword argument named settings (or **kwargs). It then
    receives the settings of the run: the row of the design table as a dictionary, with idx as the index of the
    run. A function without the argument is called with its input only, as before. The settings are what a
    training function needs to recover the data generating process, for example with create_homc
    (SynBPS.simulation.alg7_memory_process_generator) for the process with memory.

    Args:
        dataprep_function: function which takes the event-log (pandas dataframe) and returns the input data for training_function
        training_function: function which takes the input data and returns the inference table (pandas dataframe) for eval_function
        eval_function: function which takes the inference table and returns a dictionary with the evaluation metrics of the run
        store_eventlogs (bool): store the event-log and the inference table of every run in output_dir
        output_dir (str): directory of the design table, the results file and the stored event-logs
        out_file (str): name of the results file in output_dir
        design_table (str): name of the design table in output_dir, as created by make_design_table
        verbose (bool): print the progress of every run

    Returns:
        Pandas dataframe with one row per performed run: the settings, the characteristics of the event-log and the
        evaluation metrics, including the runs performed by earlier calls
    """

    # Load necessary libraries
    import pandas as pd
    import numpy as np
    import os
    import time
    import traceback
    from tqdm import tqdm


    # load the design table as df
    df = pd.read_csv(output_dir+design_table)

    # the Failure column is created by make_design_table, but a design table may come from elsewhere
    if "Failure" not in df.columns:
        df["Failure"] = 0

    # Placeholder for the results, starting with the results of earlier calls (if any)
    results = []

    if os.path.isfile(output_dir+out_file):
        results.append(pd.read_csv(output_dir+out_file))

    # the results of the runs performed so far, as one table (an empty table when there are none)
    def collect_results():
        if len(results) == 0:
            return pd.DataFrame()

        experiments = pd.concat(results, ignore_index=True)

        # a run which is performed again replaces its earlier row
        if "idx" in experiments.columns:
            experiments = experiments.drop_duplicates(subset="idx", keep="last", ignore_index=True)

        return experiments

    # Iterate over each run in the design table df
    for run in tqdm(df.index):

        """
        Retrieving settings for experiment i
        """
        # a copy, so the columns added below do not change the design table
        curr_settings = df.loc[run].copy()
        curr_settings["idx"] = run

        """
        If experiment is not previously performed
        """
        if curr_settings.Done == 0:
            if verbose==True:
                print("Run:",run)

            try:
                res_i = perform_run(run, curr_settings, dataprep_function, training_function, eval_function, store_eventlogs, output_dir, verbose)

            except Exception:
                # mark the run as failed and continue with the next run
                df.loc[run,"Failure"] = 1
                df.to_csv(output_dir + design_table, index=False)

                print("Run " + str(run) + " failed and is marked with Failure = 1 in the design table:")
                print(traceback.format_exc())
                continue

            """
            Store the results
            """

            # Mark as done in the design table
            df.loc[run,"Done"] = 1
            df.loc[run,"Failure"] = 0
            df.to_csv(output_dir + design_table, index=False)

            # Store the settings and the metrics of run i
            results.append(res_i)

            #store results
            experiments = collect_results()
            experiments.to_csv(output_dir+out_file, index=False)

    return collect_results()


def accepts_settings(function):
    """
    Whether a user function takes the settings of the run: it has a parameter named settings, or **kwargs.
    Functions whose signature cannot be read (some built-in callables) are called without the settings.
    """
    import inspect

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False

    return any(p.name == "settings" or p.kind == p.VAR_KEYWORD for p in parameters)


def call_user_function(function, argument, settings):
    """
    Calls a user function with its input, and with the settings of the run when it accepts them
    """
    if accepts_settings(function):
        return function(argument, settings=settings)

    return function(argument)


def perform_run(run, curr_settings, dataprep_function, training_function, eval_function, store_eventlogs, output_dir, verbose):
    """
    Performs a single run of the design table: generates the event-log, trains and evaluates the model.

    Args:
        run: index of the run in the design table
        curr_settings (pandas Series): the row of the design table, which is extended with the results
        the remaining arguments are those of run_experiments

    Returns:
        Pandas dataframe with one row: the settings, the characteristics of the event-log and the evaluation metrics
    """
    import pandas as pd
    import numpy as np
    import time

    # the settings of the run as given to the user functions: the row of the design table, before the
    # characteristics of the event-log are added to curr_settings below
    settings = curr_settings.to_dict()

    start_time = time.time()

    # generate the log
    from SynBPS.simulation.simulate_eventlog import generate_eventlog
    log = generate_eventlog(curr_settings=curr_settings)

    stop_time = time.time()

    # store it
    if store_eventlogs==True:
        log.to_csv(output_dir+"run_"+str(run)+"_eventlog.csv", index=False)
        if verbose==True:
            print("eventlog saved to:",output_dir+"run_"+str(run)+"_eventlog.csv")


    # store metrics from simulated log
    curr_settings["simuation_time_sec"] = stop_time - start_time
    curr_settings["num_traces"] = len(log.caseid.unique())
    curr_settings["num_events"] = len(log)

    # variants and trace lengths
    variants = []
    tracelengths = []
    tracedurations = []
    event_durations = []
    r_waitingtimes = []
    s_waitingtimes = []

    for traceid in log.caseid.unique():
        trace = log.loc[log.caseid == traceid]

        #tracelen
        tracelen = len(trace)
        tracelengths.append(tracelen)

        #trace duration
        traceduration = np.max(trace["y_acc_sum"])
        tracedurations.append(traceduration)

        #event durations
        event_duration = np.mean(trace["v_t"])
        event_durations.append(event_duration)

        #resource waiting times
        r_waitingtime = np.mean(trace["h_t"])
        r_waitingtimes.append(r_waitingtime)


        #resource waiting times
        s_waitingtime = np.mean(trace["b_t"])
        s_waitingtimes.append(s_waitingtime)

        #variant
        sequence = ""
        sequence = sequence.join(trace.activity.tolist())
        variants.append(sequence)

    # log simulated log characteristics
    n_variants = len(set(variants))
    curr_settings["num_variants"] = n_variants

    curr_settings["avg_tracelen"] = np.mean(tracelengths)
    curr_settings["min_tracelen"] = np.min(tracelengths)
    curr_settings["max_tracelen"] = np.max(tracelengths)

    curr_settings["avg_traceduration"] = np.mean(tracedurations)
    curr_settings["stdev_traceduration"] = np.std(tracedurations)
    curr_settings["min_traceduration"] = np.min(tracedurations)
    curr_settings["max_traceduration"] = np.max(tracedurations)

    curr_settings["avg_eventduration"] = np.mean(event_durations)
    curr_settings["stdev_eventduration"] = np.std(event_durations)
    curr_settings["min_eventduration"] = np.min(event_durations)
    curr_settings["max_eventduration"] = np.max(event_durations)

    curr_settings["avg_r_waitingtimes"] = np.mean(r_waitingtimes)
    curr_settings["avg_s_waitingtimes"] = np.mean(s_waitingtimes)


    """
    Prepare data for modelling (memory here refers to RAM)
    """

    input_data = call_user_function(dataprep_function, log, settings)

    """
    Train a model
    """

    ### Custom training function
    inference_test = call_user_function(training_function, input_data, settings)

    if store_eventlogs==True:
        # store inference table
        inference_test.to_csv(output_dir+"inference_test_"+str(run)+".csv", index=False)

    """
    Evaluate the model
    """

    ### Custom evaluation function
    metrics = call_user_function(eval_function, inference_test, settings)

    ### Store metrics to the in curr_settings dictionary which becomes the result table
    ### Prefixing column names is ideal for later analysis

    curr_settings["RESULT_num_events"] = len(log)

    # add evaluation metrics
    metrics = pd.DataFrame(metrics, index=[run])
    curr_settings = curr_settings.to_dict()
    curr_settings = pd.DataFrame(curr_settings, index=[run])
    res_i = pd.concat([curr_settings, metrics], axis=1)

    return res_i

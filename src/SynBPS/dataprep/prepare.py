def prefix_data(log, trainsize=0.7, category_cols=[], numeric_cols=[], dummify_time_features=False, window_position="last_k", scaling=None, verbose=False):
    """
    Turns an event-log into a prefix-log for predictive process monitoring: one observation per observed
    event of every trace, with the events seen so far as a padded sequence, and the remaining time, the
    time to the next event and the next activity as targets. The traces are split into a train and a
    test set by time. See the dataprep page of the documentation for the steps and the features.

    Args:
        log (pandas dataframe): An event-log from generate_eventlog.
        trainsize (float): Share of the dates of the log before the split date. A trace is in the train set
            when its first event is before that date.
        category_cols (list of str): Columns of the log added as one-hot features of every event, for example ["start_day"].
        numeric_cols (list of str): Columns of the log added as numeric features of every event, for example ["h_t", "s_t"].
            The features are named num_<column>.
        dummify_time_features (bool): One-hot encode dayofweek and hourofday instead of keeping them as numbers.
        window_position (str): "last_k" or "first_k": which events of a trace are kept when a trace is longer
            than the longest trace of the log. As the prefix length is the longest trace, every event is kept.
        scaling (str): None, "standardize" or "normalize". Scales the numeric features (not the one-hot
            features and not the targets) with the mean and standard deviation, or the minimum and maximum,
            of the train set. The padding of the sequences stays 0.
        verbose (bool): Print the progress.

    Returns:
        dict with
            x_train, x_test: numpy arrays (observations, prefix length, features). Each observation is the
                events of a trace up to and including one event, padded with rows of zeros in front.
            y_train, y_test: numpy arrays (observations, 1), the remaining time of the trace in seconds.
            y_t_train, y_t_test: numpy arrays (observations, 1), the time to the next event in seconds.
            y_a_train, y_a_test: numpy arrays (observations, activities), the next activity as one-hot.
            Inference_train, Inference_test: pandas dataframes with one row per observation, in the same order
                as the arrays: SEQID, caseid, log_caseid (the caseid in the event-log), num_events, prefix_number
                (the number of observed events), prefixes, prefixwindow, prefix_date (the start of the last
                observed event), distinct_events, caseduration_days, y, y_t, the columns of y_a and event_number.
            scaling: dict with the mode, the scaled columns and their parameters, to reuse or invert the scaling.
    """
    import pandas as pd
    import numpy as np
    from SynBPS.dataprep.dataprep_helperfunctions import InitialFormatting, GetFileInfo, MakeSplitCriterion, GenerateTrainData, ScaleFeatures, PadInputs, CaseData, GetCaseStats, SplitAndReshape

    #error handling
    for column in list(category_cols) + list(numeric_cols):
        if column not in log.columns:
            raise ValueError("The column " + str(column) + " is not in the event-log. Change category_cols or numeric_cols.")

    if window_position not in ["last_k", "first_k"]:
        raise ValueError("window_position must be last_k or first_k, but is " + str(window_position) + ". Change window_position.")

    if scaling not in [None, "standardize", "normalize"]:
        raise ValueError("scaling must be None, standardize or normalize, but is " + str(scaling) + ". Change scaling.")

    # select relevant columns
    extra_cols = [c for c in list(category_cols) + list(numeric_cols) if c not in ['caseid', 'activity', 'activity_no', 'start_datetime', 'end_datetime']]
    log = log[['caseid', 'activity', 'activity_no', 'start_datetime', 'end_datetime'] + extra_cols]

    # rename
    log = log.rename({"caseid":"id",
                 "start_datetime":"time",
                 "activity":"event"}, axis='columns')

    #format: drop traces with one event, renumber the cases from 1 and sort by time
    df = InitialFormatting(log, maxcases=1000000, dateformat="%Y-%m-%d %H:%M:%S", verbose=False)
    df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.index = list(range(0,len(df)))


    # get max trace length in the log
    max_length = GetFileInfo(df)

    # temporal split based on events or cases
    splitmode =  ["event", "case"][0]

    if verbose == True:
        print("mode:",splitmode)
        print("=======================================")

    # make split criteria
    split_criterion = MakeSplitCriterion(df, trainsize=trainsize, mode=splitmode)

    # generate the train data: target etc.
    X, y, y_a, y_t, cases, y_a_varnames = GenerateTrainData(df,
                                                          category_cols=list(category_cols),
                                                          numeric_cols=list(numeric_cols),
                                                          dateformat = "%Y-%m-%d %H:%M:%S",
                                                          droplastev=True,
                                                          drop_end_target=True,
                                                          get_activity_target=True,
                                                          get_case_features = True,
                                                          dummify_time_features = dummify_time_features,
                                                          max_prefix_length = max_length,
                                                          window_position=window_position)

    # pad the data
    padded_df = PadInputs(cases, 
                      X, 
                      max_prefix_len=max_length)
    
    # scale the numeric features (not the caseid and not the one-hot columns) with the parameters of the train set
    numeric_features = [c for c in X.columns if c != "caseid" and X[c].dtype != bool]
    train_caseids = split_criterion.loc[split_criterion["trainset"] == True, "caseid"].tolist()
    padded_df, scaling_info = ScaleFeatures(padded_df, train_caseids, numeric_features, mode=scaling, verbose=verbose)

    # get case data
    Case_Data = CaseData(df)

    # get case stats
    CaseStats = GetCaseStats(df.rename({"activity_no":"event_number"},axis=1),
                         padded_df,
                         Case_Data,
                         y_t,
                         y_a,
                         y,
                         prefixwindow=max_length,
                         dateformat="%Y-%m-%d %H:%M:%S",
                         drop_last_ev=True)

    # split and reshape data for RNNs
    X_train, X_test, y_t_train, y_t_test, y_a_train, y_a_test, y_train, y_test = SplitAndReshape(padded_df,
                                                                                             y_a,
                                                                                             y_t,
                                                                                             y,
                                                                                             split_criterion,
                                                                                             prefixlength=max_length)
    # make inference tables
    Inference = pd.merge(left=CaseStats, right=split_criterion, on="caseid",how="left")

    Inference_train = Inference.loc[Inference.trainset==True].drop("trainset",axis=1)
    Inference_test = Inference.loc[Inference.trainset==False].drop("trainset",axis=1)
    if verbose == True:
        print("Inference train:",len(Inference_train))
        print("Inference test: ",len(Inference_test))

    #collect all datasets
    Input_data={"x_train":X_train,
            "x_test":X_test,
            "y_train":y_train,
            "y_test":y_test,
            "y_a_train":y_a_train,
            "y_a_test":y_a_test,
            "y_t_train":y_t_train,
            "y_t_test":y_t_test,
            "Inference_train":Inference_train,
            "Inference_test":Inference_test,
            "scaling":scaling_info}

    return Input_data

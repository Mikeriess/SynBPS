def split_cases(log, trainsize=0.7, split_by="cases", spanning_cases="train", split_date=None, drop_single_event_cases=True):
    """
    Assigns the cases of an event-log to the train set, the test set or neither, by time. The same rules are
    used by prefix_data, which calls this function; the table can also be computed on its own and given to
    other pipelines, or to prefix_data as case_sets, so that every model is evaluated on the same cases.

    Rules, in this order:
        1. Cases with a single event are dropped when drop_single_event_cases is True.
        2. The split date is split_date when given. Otherwise, with split_by "cases", it is the start of the
           first case in the test set when the remaining cases are ordered by their start, such that a share
           of trainsize of them is in the train set (as close as possible, when several cases start at the
           same time). With split_by "date", it is the start of the earliest event plus trainsize of the days
           between the earliest and the latest event (the rule up to 1.1.9).
        3. A case is in the train set when its first event starts before the split date, and in the test set
           otherwise.
        4. A train case whose last event starts at or after the split date spans the split. With spanning_cases
           "train" it stays in the train set, with "drop" it is dropped.

    Args:
        log (pandas dataframe): An event-log from generate_eventlog, with the columns caseid, activity_no and start_datetime.
        trainsize (float): Share of the cases (split_by "cases") or of the days of the log (split_by "date") in the train set, between 0 and 1.
        split_by (str): "cases" or "date", see the rules.
        spanning_cases (str): "train" or "drop", see the rules.
        split_date: A timestamp or a string such as "2024-03-01", which replaces the computed split date.
        drop_single_event_cases (bool): Drop the cases with a single event. Such a case gives no observation
            in prefix_data unless include_end is True.

    Returns:
        pandas dataframe with one row per case of the log, in the order of the log:
            log_caseid: the caseid of the log
            num_events: number of events of the case
            start: start of the first event
            end: start of the last event
            set: "train", "test" or "dropped"
            dropped_reason: "single event", "spans the split date" or "" (empty when the case is kept)
            split_date: the split date, the same in every row
    """
    import pandas as pd
    import numpy as np

    #error handling
    for column in ["caseid", "activity_no", "start_datetime"]:
        if column not in log.columns:
            raise ValueError("The column " + column + " is not in the event-log. Use an event-log from generate_eventlog.")

    if split_by not in ["cases", "date"]:
        raise ValueError("split_by must be cases or date, but is " + str(split_by) + ". Change split_by.")

    if spanning_cases not in ["train", "drop"]:
        raise ValueError("spanning_cases must be train or drop, but is " + str(spanning_cases) + ". Change spanning_cases.")

    if split_date is None and not (0 < trainsize < 1):
        raise ValueError("trainsize must be between 0 and 1 (both excluded), but is " + str(trainsize) + ". Change trainsize.")

    if split_date is not None:
        try:
            split_date = pd.Timestamp(split_date)
        except (TypeError, ValueError):
            raise ValueError("split_date must be a timestamp or a date string, but is " + str(split_date) + ". Change split_date.")

    #one row per case: number of events, start of the first and of the last event
    events = log[["caseid", "activity_no", "start_datetime"]].copy()
    events["start_datetime"] = pd.to_datetime(events["start_datetime"])
    events = events.sort_values(["caseid", "activity_no"], kind="stable")

    cases = events.groupby("caseid", sort=False).agg(num_events=("activity_no", "size"),
                                                     start=("start_datetime", "first"),
                                                     end=("start_datetime", "last")).reset_index()
    cases = cases.rename(columns={"caseid":"log_caseid"})

    #the order of the log
    order = {c: i for i, c in enumerate(log["caseid"].unique())}
    cases = cases.sort_values("log_caseid", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)

    cases["set"] = "train"
    cases["dropped_reason"] = ""

    #rule 1: cases with a single event
    if drop_single_event_cases:
        single = cases["num_events"] == 1
        cases.loc[single, "set"] = "dropped"
        cases.loc[single, "dropped_reason"] = "single event"

    kept = cases["set"] != "dropped"

    if kept.sum() < 2:
        raise ValueError("The event-log has fewer than 2 cases to split. Generate more traces, or set drop_single_event_cases to False.")

    #rule 2: the split date
    if split_date is None:
        if split_by == "cases":
            #the start of the first test case, when the cases are ordered by their start. As the cases which start
            #at the split date are in the test set, several cases with the same start (for example at the opening
            #hour) cannot be separated: the split date is then placed before or after them, whichever gives a
            #share of train cases closer to trainsize
            starts = np.sort(cases.loc[kept, "start"].values)
            n_train = int(round(trainsize * len(starts)))
            n_train = min(max(n_train, 1), len(starts) - 1)
            
            candidates = [starts[n_train]]
            later = starts[starts > starts[n_train]]
            if len(later) > 0:
                candidates.append(later[0])
            
            counts = [int((starts < d).sum()) for d in candidates]
            valid = [(abs(c - n_train), i) for i, c in enumerate(counts) if 0 < c < len(starts)]
            if len(valid) == 0:
                raise ValueError("The cases cannot be split by their start: too many cases start at the same time. Give a split_date, or use split_by date.")
            split_date = pd.Timestamp(candidates[min(valid)[1]])

        if split_by == "date":
            #the earliest event plus trainsize of the days between the earliest and the latest event
            kept_events = events.loc[events["caseid"].isin(cases.loc[kept, "log_caseid"]), "start_datetime"]
            earliest = kept_events.min()
            latest = kept_events.max()
            n_dates = (latest - earliest).days + 1
            split_date = earliest + pd.Timedelta(days=int(round(n_dates * trainsize)))

    #rule 3: a case is in the train set when its first event starts before the split date
    cases.loc[kept & (cases["start"] >= split_date), "set"] = "test"

    #rule 4: train cases whose last event starts at or after the split date
    spanning = kept & (cases["start"] < split_date) & (cases["end"] >= split_date)
    if spanning_cases == "drop":
        cases.loc[spanning, "set"] = "dropped"
        cases.loc[spanning, "dropped_reason"] = "spans the split date"

    if (cases["set"] == "train").sum() == 0 or (cases["set"] == "test").sum() == 0:
        raise ValueError("The split leaves no cases in the train set or in the test set. Change trainsize, split_date or spanning_cases.")

    cases["split_date"] = split_date

    return cases


def prefix_data(log, trainsize=0.7, split_by="cases", spanning_cases="train", split_date=None, case_sets=None,
                drop_single_event_cases=True, include_end=False,
                category_cols=[], numeric_cols=[], dummify_time_features=False, window_position="last_k", scaling=None, verbose=False):
    """
    Turns an event-log into a prefix-log for predictive process monitoring: one observation per observed
    event of every trace, with the events seen so far as a padded sequence, and the remaining time, the
    time to the next event and the next activity as targets. The traces are split into a train and a
    test set by time. See the dataprep page of the documentation for the steps and the features.

    Args:
        log (pandas dataframe): An event-log from generate_eventlog.
        trainsize (float): Share of the cases in the train set (between 0 and 1). With split_by "date" it is
            the share of the days of the log instead.
        split_by (str): "cases": the split date is chosen such that a share of trainsize of the cases is in
            the train set. "date": the split date is trainsize of the way from the earliest to the latest event.
        spanning_cases (str): What happens to a train case whose last event starts at or after the split date.
            "train": it stays in the train set. "drop": it is left out.
        split_date: A timestamp or date string which replaces the computed split date; trainsize and split_by are then not used.
        case_sets (pandas dataframe): The table from split_cases, which replaces the split computed here. Only the
            columns log_caseid and set are used. trainsize, split_by, spanning_cases, split_date and
            drop_single_event_cases are then not used.
        drop_single_event_cases (bool): Drop the cases with a single event. Such a case only gives an observation when include_end is True.
        include_end (bool): Keep the last event of every case as an observation. Its next activity is END, which
            is then a column of y_a, and its remaining time and time to the next event are 0. By default the last
            event is dropped and y_a has no END column.
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
            y_a_train, y_a_test: numpy arrays (observations, activities), the next activity as one-hot
                (activities plus END with include_end).
            Inference_train, Inference_test: pandas dataframes with one row per observation, in the same order
                as the arrays: SEQID, caseid, log_caseid (the caseid in the event-log), num_events, prefix_number
                (the number of observed events), prefixes, prefixwindow, prefix_date (the start of the last
                observed event), distinct_events, caseduration_days, y, y_t, the columns of y_a and event_number.
            cases: the table of split_cases: one row per case of the log with its set (train, test or dropped).
            split_date: the split date (pandas Timestamp).
            scaling: dict with the mode, the scaled columns and their parameters, to reuse or invert the scaling.
    """
    import warnings
    import pandas as pd
    import numpy as np
    from SynBPS.dataprep.dataprep_helperfunctions import InitialFormatting, GetFileInfo, GenerateTrainData, ScaleFeatures, PadInputs, CaseData, GetCaseStats, SplitAndReshape

    #error handling
    for column in list(category_cols) + list(numeric_cols):
        if column not in log.columns:
            raise ValueError("The column " + str(column) + " is not in the event-log. Change category_cols or numeric_cols.")

    if window_position not in ["last_k", "first_k"]:
        raise ValueError("window_position must be last_k or first_k, but is " + str(window_position) + ". Change window_position.")

    if scaling not in [None, "standardize", "normalize"]:
        raise ValueError("scaling must be None, standardize or normalize, but is " + str(scaling) + ". Change scaling.")

    if not include_end and not drop_single_event_cases:
        warnings.warn("Cases with a single event give no observation when include_end is False: their only event is the last event, which is dropped. Set include_end to True to keep them.")

    """
    Train and test split of the cases
    """
    if case_sets is None:
        cases = split_cases(log, trainsize=trainsize, split_by=split_by, spanning_cases=spanning_cases,
                            split_date=split_date, drop_single_event_cases=drop_single_event_cases)
    else:
        for column in ["log_caseid", "set"]:
            if column not in case_sets.columns:
                raise ValueError("case_sets must have the columns log_caseid and set, as returned by split_cases.")
        if not set(case_sets["set"].unique()) <= {"train", "test", "dropped"}:
            raise ValueError("The column set of case_sets must contain train, test or dropped only.")
        if set(case_sets["log_caseid"].astype(str)) != set(log["caseid"].astype(str)):
            raise ValueError("case_sets must have one row for every case of the event-log. Use split_cases on the same event-log.")
        cases = case_sets.copy()
        cases["log_caseid"] = cases["log_caseid"].astype(str)
        if "split_date" not in cases.columns:
            cases["split_date"] = pd.NaT

    split_date = cases["split_date"].iloc[0]

    #only the cases of the train and test set
    kept_cases = cases.loc[cases["set"] != "dropped", "log_caseid"].astype(str)
    log = log.loc[log["caseid"].astype(str).isin(kept_cases)]

    if verbose == True:
        print("split date:", split_date)
        print("cases:", cases["set"].value_counts().to_dict())
        print("=======================================")

    # select relevant columns
    extra_cols = [c for c in list(category_cols) + list(numeric_cols) if c not in ['caseid', 'activity', 'activity_no', 'start_datetime', 'end_datetime']]
    log = log[['caseid', 'activity', 'activity_no', 'start_datetime', 'end_datetime'] + extra_cols]

    # rename
    log = log.rename({"caseid":"id",
                 "start_datetime":"time",
                 "activity":"event"}, axis='columns')

    #format: renumber the cases from 1 and sort by time (the cases were selected above)
    df = InitialFormatting(log, maxcases=1000000, dateformat="%Y-%m-%d %H:%M:%S", drop_single_event_cases=False, verbose=False)
    df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df['time_parsed'] = pd.to_datetime(df['time'])
    df.index = list(range(0,len(df)))

    # get max trace length in the log
    max_length = GetFileInfo(df)

    # the split criterion: the set of every (renumbered) case
    split_criterion = df[["id", "log_caseid"]].drop_duplicates("id").rename(columns={"id":"caseid"})
    split_criterion["log_caseid"] = split_criterion["log_caseid"].astype(str)
    split_criterion = pd.merge(left=split_criterion, right=cases[["log_caseid", "set"]], on="log_caseid", how="left")
    split_criterion["trainset"] = split_criterion["set"] == "train"
    split_criterion = split_criterion[["caseid", "trainset"]].reset_index(drop=True)

    # generate the train data: target etc.
    X, y, y_a, y_t, caseids, y_a_varnames = GenerateTrainData(df,
                                                          category_cols=list(category_cols),
                                                          numeric_cols=list(numeric_cols),
                                                          dateformat = "%Y-%m-%d %H:%M:%S",
                                                          droplastev=not include_end,
                                                          drop_end_target=not include_end,
                                                          get_activity_target=True,
                                                          get_case_features = True,
                                                          dummify_time_features = dummify_time_features,
                                                          max_prefix_length = max_length,
                                                          window_position=window_position)

    # pad the data
    padded_df = PadInputs(caseids,
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
                         drop_last_ev=not include_end)

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
            "cases":cases,
            "split_date":split_date,
            "scaling":scaling_info}

    return Input_data

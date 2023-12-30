
def prefix_data(log, verbose=False):
    import pandas as pd
    import numpy as np
    from SynBPS.dataprep.dataprep_helperfunctions import InitialFormatting, GetFileInfo, MakeSplitCriterion, GenerateTrainData, PadInputs, CaseData, GetCaseStats, SplitAndReshape
    
    # select relevant columns
    log = log[['caseid', 'activity', 'activity_no', 'start_datetime','end_datetime']]
    
    # rename
    log = log.rename({"caseid":"id",
                 "start_datetime":"time",
                 "activity":"event"}, axis='columns')

    #format
    df = InitialFormatting(log, maxcases=1000000, dateformat="%Y-%m-%d %H:%M:%S")#"%Y-%M-%d %H:%m:%ss")
    df.index = list(range(0,len(df)))

    #convert back ??? lol why not
    df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # get max trace length in the log
    max_length = GetFileInfo(df)
    
    splitmode = "event"
    print("mode:",splitmode)
    print("=======================================")
    
    # make split criteria
    split_criterion = MakeSplitCriterion(df, trainsize=0.5, mode=splitmode) # "event", "case"
    
    # generate the train data: target etc.
    X, y, y_a, y_t, cases, y_a_varnames = GenerateTrainData(df,
                                                          category_cols=[], #"start_day"
                                                          numeric_cols=[],
                                                          dateformat = "%Y-%m-%d %H:%M:%S",
                                                          droplastev=True,
                                                          drop_end_target=True,
                                                          get_activity_target=True,
                                                          get_case_features = True,
                                                          dummify_time_features = False, 
                                                          max_prefix_length = max_length,
                                                          window_position="last_k")
    
    # pad the data
    padded_df = PadInputs(cases, 
                      X, 
                      max_prefix_len=max_length, 
                      standardize=True)

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
            "Inference_test":Inference_test}
    
    return Input_data
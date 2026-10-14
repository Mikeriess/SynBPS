# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:43:27 2019

@author: mikeriess
"""
import pandas as pd
import numpy as np


def InitialFormatting(df, maxcases, dateformat, drop_single_event_cases=True, verbose=False):
    """
    Keeps the first maxcases cases, drops the cases with a single event (when drop_single_event_cases is True),
    keeps the caseid of the log in log_caseid, renumbers the cases 1, 2, 3, ... and sorts the events by case and time
    """
    import pandas as pd
    
    #Work on a subset:
    casestoload = df["id"].unique().tolist()[0:maxcases]
    df = df.loc[df["id"].isin(casestoload)].copy()
    
    if drop_single_event_cases:
        if verbose == True:
            # find cases to drop due to length
            print("Cases before dropping len=1:",len(casestoload),"cases",len(df),"rows")
        
        # Make list of cases above length 1
        sizes = df.groupby('id').size()
        keepers = sizes[sizes > 1].index.tolist()
        
        # Drop cases with only one event:
        df = df.loc[df["id"].isin(keepers)].copy()
        
        if verbose == True:
            print("Cases after dropping len=1:",len(keepers),"cases",len(df),"rows")
    
    #keep the caseid of the log, as the cases get new consecutive ids below
    df["log_caseid"] = df["id"]
    
    #Sort the dataframe by time aftewards
    df['parsed_date'] = pd.to_datetime(df.time, format = dateformat, exact = True)
    
    ##########################################################################
    if verbose == True:
        print("Sorting by id, date (chronological order)")
    #generate new ID column:
    df["id"] = df["id"].astype('category').cat.codes.astype('int32')
    
    # Ensure ID starts at 1
    if min(df.id) == 0:
        df["id"] = df.id + 1
    
    # Sort the DF baed on caseid, and the date of the event
    df = df.sort_values(['id',"parsed_date"], ascending=[True, True]) 
    
    df = df.drop("parsed_date",axis=1)
    return df


def GetFileInfo(df, verbose=False):
    if verbose == True:
        print("Number of cases in log:",len(df["id"].unique()))
    import numpy as np
    import pandas as pd
    
    #Get the maximal trace length, for determining prefix length
    max_length = np.max(df['id'].value_counts())
    if verbose == True:
        print("longest trace is:",max_length)
    
    #Look at the time format:
    if verbose == True:
        print("Time format:",df["time"].loc[0])
        print("Std. format: %Y-%m-%d %H:%M:%S")
        print(df.head())
    return max_length




def MakeSplitCriterion(df, trainsize=0.8, mode="event", verbose=False):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time as tm
    
    def datetime_range(start=None, end=None):
        span = end - start
        for i in range(span.days + 1):
            yield start + timedelta(days=i)
 
    #Parse date
    df["time_parsed"] = pd.to_datetime(df["time"])
    
    #Get min max dates:
    earliest_date = min(df["time_parsed"])
    lastest_date = max(df["time_parsed"])
    
    #Find the date to divide on:
    dates = list(datetime_range(start=earliest_date, end=lastest_date))
    n_dates = len(dates)
    splitpoint = n_dates*trainsize
    splitpoint = int(np.round(splitpoint,decimals=0))
    
    dividing_date = dates[splitpoint]
    dividing_date = dividing_date

    if verbose == True:
        print("=======================================")
        print("Log starts at:",earliest_date)
        print("Last event starts at:",lastest_date)
        print("Train-test split happens at:",dividing_date)
        print("=======================================")

        
    if mode=="event":
        """
            Here we simply divide by date of the event,
            and disregard that a case could be in both train and test set
            this way
        """
        
        df["trainset"] = df["time_parsed"] < dividing_date
        df["trainset"].value_counts()
        
        split_criterion = df[["id","trainset"]]
        split_criterion = split_criterion.rename(columns={'id':'caseid',
                                'trainset':'trainset'}, inplace=False)
        
        split_criterion = split_criterion.reset_index(drop=True)
        split_criterion = split_criterion.drop_duplicates(subset="caseid",keep="first")
        
        if verbose == True:
            print(len(split_criterion["caseid"].unique().tolist()))
            print(len(split_criterion))
            print(np.sum(df["trainset"]*1))
            print("=======================================")
        
    if mode=="case":
        """
            Here we remove all cases that are in both train and test set
        """
        
        # For every case, verify if it has both True & False events
        # If it has, drop that case ID
        # And remember to print it
        
        df["trainset"] = df["time_parsed"] < dividing_date
        df["trainset"].value_counts()
        
        split_criterion = df[["id","trainset"]]
        split_criterion = split_criterion.rename(columns={'id':'caseid',
                                'trainset':'trainset'}, inplace=False)
        
        split_criterion = split_criterion.reset_index(drop=True)
        
        #Groupby and get count of every unique value per case id
        validation = pd.DataFrame(split_criterion.groupby('caseid').trainset.nunique())
        validation["caseid"] = validation.index
        
        #If a caseid has both true and false within it (count == 2),
        #it should be dropped.
        if verbose == True:
            print("=======================================")
            print("Dropping cases that have events in both train + testsets:")
            print("=======================================")
            print("Cases before dropping:",len(validation["trainset"]))
        validation["keep"] = validation["trainset"] == 1
        validation = validation.loc[validation["keep"]==True]
        if verbose == True:
            print("Cases after dropping:",len(validation["trainset"]))
        
        #list of caseids to keep
        ids_keep = validation["caseid"]
        
        #drop those thet should not be kept
        if verbose == True:
            print("Total events before:",len(split_criterion))
        split_criterion = split_criterion.loc[split_criterion["caseid"].isin(ids_keep)]
        if verbose == True:
            print("Total events after:",len(split_criterion))
        split_criterion = split_criterion.drop_duplicates(subset="caseid",keep="first")
        if verbose == True:
            print("=======================================")
            print(len(split_criterion))
            print(np.sum(split_criterion["trainset"]*1))
    return split_criterion










def GenerateTrainData(df,
                      category_cols=[],
                      numeric_cols=[],
                      dateformat = "%Y-%m-%d %H:%M:%S",
                      droplastev=True,
                      drop_end_target=True,
                      get_activity_target=True,
                      get_case_features = True,
                      dummify_time_features = True, 
                      max_prefix_length = 2,
                      window_position="last_k",
                      verbose=False):
    import pandas as pd
    import numpy as np
    
    #Make copy of df, so the table of the caller is not changed
    data = df.copy()
    
    # List all cases, in the order of the table (sorted by case and time)
    cases = data["id"].unique().tolist()
    
    # Make new event ids: the number of the event in its case, 1 for the first event
    data["eventid"] = data.groupby("id").cumcount() + 1
    
    """
    #######################################################################
    PREFIX: the window of events of every case
    #######################################################################
    """
    case_length = data.groupby("id")["eventid"].transform("max")
    
    #determine whether to start in the beginning or end of trace        
    if window_position == "last_k":
        #if the max prefix len is smaller than the actual trace, take the K last events, otherwise the full trace
        window_start = np.where(max_prefix_length < case_length, case_length - max_prefix_length, 1)
        window_stop = case_length
    
    if window_position == "first_k":
        #if the max prefix len is smaller than the actual trace, take the K first events, otherwise the full trace
        window_start = 1
        window_stop = np.where(max_prefix_length < case_length, max_prefix_length, case_length)
    
    #Prefix capability: Subset the window of events of every case (the data to be dummified also follows the prefix convention)
    datasub = data.loc[(data["eventid"] >= window_start) & (data["eventid"] <= window_stop)].copy()
    datasub.index = datasub["eventid"]
    
    """
    #######################################################################
    Time features for each event, relative to the window of its case
    #######################################################################
    """
    t = pd.to_datetime(datasub["time"], format=dateformat)
    by_case = t.groupby(datasub["id"].values)
    
    #first and last event of the window, previous and next event
    t_start = by_case.transform("first")
    t_stop = by_case.transform("last")
    t_last = by_case.shift(1)
    t_next = by_case.shift(-1)
    
    #the next activity, END for the last event of the window
    next_activity = datasub["event"].groupby(datasub["id"].values).shift(-1).fillna("END")
    
    out = pd.DataFrame({
        "caseid": datasub["id"].astype(str).values,
        "event_number": datasub["eventid"].values,
        "event_activity": datasub["event"].values,
        #time in secs since midnight
        "timesincemidnight": (t - t.dt.normalize()).dt.total_seconds().values,
        #monday = 1
        "dayofweek": (t.dt.weekday + 1).values,
        #hour of day
        "hourofday": t.dt.hour.values,
        #Time since start in seconds
        "timesincestart": (t - t_start).dt.total_seconds().values,
        #Time since last event in seconds, 0 for the first event
        "timesincelastev": (t - t_last).dt.total_seconds().fillna(0).values,
        #Time until finishtime
        "y_timetofinish": (t_stop - t).dt.total_seconds().values,
        #Time to next event, 0 for the last event
        "y_timetonextev": (t_next - t).dt.total_seconds().fillna(0).values,
        "next_activity": next_activity.values,
        #Make a marker for dropping last step where remaining time (y) = 0
        "drop": t_next.isna().astype(int).values,
    })
      
    if verbose == True:
        print("============================")
        print("Post-processing:")
        print("============================")
    #####################################
    # One-hot encoding
    
    #store original labels:
    
    #convert event into numerical codes
    out['event_activity'] = out['event_activity'].astype('category')
    out['event_activity'] = out['event_activity'].cat.codes
    out['event_activity'] = out['event_activity']+1
    
    #do the same for the subset of categorical features
    datasub['event_activity'] = datasub['event'].astype('category')
    datasub['event_activity'] = datasub['event_activity'].cat.codes
    datasub['event_activity'] = datasub['event_activity']+1
        
    if get_activity_target == True:
        #dummify next event variable
        y_a = pd.get_dummies(out["next_activity"], prefix="y_a_t1")
        y_a = y_a.reset_index(drop=True)    
        #generate list of original varnames
        y_a_varnames = y_a.columns.tolist()
    
        #do it all again, but with renamed activity names
        #convert event into numerical codes
        out['next_activity'] = out['next_activity'].astype('category')
        out['next_activity'] = out['next_activity'].cat.codes
        out['next_activity'] = out['next_activity']+1
        
        #dummify next event variable
        y_a = pd.get_dummies(out["next_activity"], prefix="y_a_t1")
        y_a = y_a.reset_index(drop=True)   
        
        #generate list of new original varnames
        y_a_new_varnames = y_a.columns.tolist()
        
        #add it into the output table
        out = out.reset_index(drop=True)
        out = pd.concat([out, y_a], axis=1)
    
    #No matter what, we always want the activity to be a feature:
    Dummies = pd.get_dummies(out["event_activity"].astype('str'))
    Dummies = Dummies.reset_index(drop=True)
    Dummies = Dummies.add_prefix('ev_a_t0_')
    out = out.reset_index(drop=True)
    dummycols = Dummies.columns.tolist()
    out = pd.concat([out,Dummies],axis=1)
    out = out.drop("event_activity",axis=1)
    
    
    if get_case_features == True:    
        if len(category_cols) > 0:
            
            dummylist = category_cols
            if verbose == True:
                print("\nDummification of",dummylist)
    
            Dummies = pd.get_dummies(datasub[dummylist])
            Dummies = Dummies.reset_index(drop=True)
            out = out.reset_index(drop=True)
            dummycols = Dummies.columns.tolist()
            out = pd.concat([out,Dummies],axis=1)
        
        if len(numeric_cols) > 0:
            #add numerical features:
            if verbose == True:
                print("Adding numerical features:",numeric_cols)
            numerics = datasub[numeric_cols]
            numerics = numerics.reset_index(drop=True)
            numerics = numerics.add_prefix('num_')
            out = out.reset_index(drop=True)
            out = pd.concat([out,numerics],axis=1)
    
    
    if dummify_time_features == True:
        features = ["dayofweek","hourofday"]
        if verbose == True:
            print("Dummification of time features",features)
        #as object columns, so that get_dummies encodes the integer values (sorted as numbers)
        sysdummies = out[features].astype(object)
        sysdummies = pd.get_dummies(sysdummies,prefix="t_")
        sysdummies = sysdummies.reset_index(drop=True)
        out = out.drop(features,axis=1)
        out = out.reset_index(drop=True)
        out = pd.concat([out, sysdummies],axis=1)
    
    #Remove last event in each case
    if droplastev==True:
        if verbose == True:
            print("\ndropping last event from each case")
            print("before:",len(out))
        out = out.loc[out["drop"] != 1]
        out = out.drop("drop",axis=1)
        if verbose == True:
            print("after:",len(out))
            print("data in X is the",max_prefix_length,"last events, excluding the final event")
    
    #####################################
    # Separate outputs
    #####################################
    
    if verbose == True:
        print("\ndropping vars:")
    
    # Generate the next_activity target vector
    y_a = out[y_a_new_varnames].copy()
    y_a["caseid"] = out["caseid"]
    y_a["event_number"] = out["event_number"]
    
    
    if drop_end_target==True:
        """ NOT SURE IF THIS SHOULD BE DONE FOR THE TAX MODEL
            as this would signal the end of a trace..
        """
        dropme = ['y_a_t1_1']#before it was: ["y_a_t1_END"]
        if verbose == True:
            print("dropping last event category from y_a:",dropme)
        y_a = y_a.drop(dropme,axis=1) #drop indicator that it is the last event
    
    
    # Generate the time to next activity target vector
    y_t = out["y_timetonextev"]
    #y_t["caseid"] = out["caseid"]
    #y_t["event"] = out["event"]
    
    y = out["y_timetofinish"]
    #y["caseid"] = out["caseid"]
    #y["event"] = out["event"]
    
    #Drop everything that is not for the model to see during training
    drops = ["y_timetofinish","y_timetonextev","next_activity"] + y_a_new_varnames
    if verbose == True:
        print("dropping vars from X: ",drops)
    
    #remove next activity maker to avoid peeking into the future
    X = out.drop(drops,axis=1)
    
    #Reset indexes:
    X = X.reset_index(drop=True)
    y_a = y_a.reset_index(drop=True)
    y_t = y_t.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    #####################################
    # Output stuff
    """
    datasub #input data used, with desired prefix
    X            # X features
    y_a          # y next event type
    y_t          # y time to next event
    y            # y remaining time at time t
    cases        # case ids for generating stats and back-linking results
    y_a_varnames # original varnames for y_a before renaming
    """
    return X, y, y_a, y_t, cases, y_a_varnames



def PadInputs(caseids, df, max_prefix_len=3, verbose=False):    
    """
    One padded sequence per observed event: the rows of the events of its case up to and including it, with rows of
    zeros in front, so that every sequence has max_prefix_len rows. The sequences are stacked in the order of the
    cases and the events, with the columns of df, the caseid and a SEQID <caseid>_<event_number> per sequence
    """
    import pandas as pd
    import numpy as np
    
    cols = df.columns.tolist()
    caseid_col = cols.index("caseid")
    event_col = cols.index("event_number")
    
    values = df.values
    caseid_values = df["caseid"].astype(str).values
    
    blocks = []
    seqids = []
    
    #loop through the cases
    for i in caseids:
        
        #Get only events from case i
        rows = values[caseid_values == str(i)]
        events = rows[:, event_col].astype(float)
        
        for j in pd.unique(events):
            
            #Get current timestep, and all earlier timesteps
            EV = rows[events < j + 1]
            
            #Figure out how many rows to pad
            rowstoadd = max_prefix_len - len(EV)
            
            ### Padding: pre-event-padding ###
            zeros = np.zeros((rowstoadd, len(cols)))
            
            #Add the zeros before the actual data
            block = np.vstack([zeros, EV]).astype(object)
            block[:, caseid_col] = str(i)
            
            blocks.append(block)
            seqids.append(str(i) + "_" + str(int(j)))
    
    dataset = pd.DataFrame(np.vstack(blocks), columns=cols)
    
    #Set an ID for the sequence
    dataset["SEQID"] = np.repeat(np.array(seqids, dtype=object), max_prefix_len)
    
    if verbose == True:
        print("\n\nOutput length:",len(dataset))
    return dataset



def ScaleFeatures(padded_df, train_caseids, columns, mode=None, verbose=False):
    """
    Scales the numeric features of the padded prefix table, with the parameters of the train set only.
    The padding rows (event_number 0) stay 0, and the one-hot columns are not scaled.
    
    Parameters
    ----------
    padded_df : the padded prefix table from PadInputs, with the columns caseid, SEQID and event_number
    train_caseids : the caseids (as in padded_df) of the train set, which the scaling parameters are computed from
    columns : the columns to scale (the numeric features)
    mode : None (no scaling), "standardize" ((x - mean) / std) or "normalize" ((x - min) / (max - min))
    
    Returns
    -------
    padded_df : the scaled table (a copy)
    scaling : dict with the mode, the scaled columns and the parameters per column (mean and std, or min and max)
    """
    import pandas as pd
    import numpy as np
    
    #error handling
    if mode not in [None, "standardize", "normalize"]:
        raise ValueError("scaling must be None, standardize or normalize, but is " + str(mode) + ". Change scaling.")
    
    padded_df = padded_df.copy()
    
    scaling = {"mode":mode, "columns":list(columns), "parameters":{}}
    
    if mode is None:
        return padded_df, scaling
    
    #the rows of observed events (not padding), and those of them in the train set
    observed = padded_df["event_number"].astype(float) > 0
    train = observed & padded_df["caseid"].astype(str).isin([str(c) for c in train_caseids])
    
    for c in columns:
        values = padded_df.loc[observed, c].astype(float)
        train_values = padded_df.loc[train, c].astype(float)
        
        if mode == "standardize":
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=0))
            #a constant feature is centered only
            scaled = (values - mean) / (std if std > 0 else 1.0)
            scaling["parameters"][c] = {"mean":mean, "std":std}
        
        if mode == "normalize":
            low = float(train_values.min())
            high = float(train_values.max())
            #a constant feature becomes 0
            scaled = (values - low) / ((high - low) if high > low else 1.0)
            scaling["parameters"][c] = {"min":low, "max":high}
        
        padded_df[c] = padded_df[c].astype(float)
        padded_df.loc[observed, c] = scaled
    
    if verbose == True:
        print("Scaled features (" + str(mode) + "):", list(columns))
    
    return padded_df, scaling


def CaseData(df, verbose=False):
    if verbose == True:
        print("Generating case data")
    # step 0: Get case aggregate stats:
    import pandas as pd
    CaseData = pd.DataFrame(df['id'].value_counts())
    CaseData.rename(columns={"count":"num_events"}, inplace=True)
    CaseData["caseid"] = CaseData.index
    CaseData.sort_values('caseid', inplace=True)
    
    distinct_events = df.groupby('id').event.nunique()
    CaseData["distinct_events"] = distinct_events
    
    #the caseid of the event-log, before the cases were renumbered
    if "log_caseid" in df.columns:
        CaseData["log_caseid"] = df.groupby('id').log_caseid.first()
    else:
        CaseData["log_caseid"] = CaseData["caseid"]
    
    mindate = df[df.groupby('id').time.transform('min') == df['time']]
    mindate = mindate.drop_duplicates(subset="id",keep="first")[["id","time"]]
    mindate.rename(columns={"time":"start"}, inplace=True)
    
    maxdate = df[df.groupby('id').time.transform('max') == df['time']]
    maxdate = maxdate.drop_duplicates(subset="id",keep="first")[["id","time"]]
    maxdate.rename(columns={"time":"stop"}, inplace=True)
    
    Dates = pd.merge(left=mindate,right=maxdate, on="id")
    Dates["start"] = pd.to_datetime(Dates["start"])
    Dates["stop"] = pd.to_datetime(Dates["stop"])
    
    import datetime as dt
    Dates["caseduration_days"] = (Dates['stop'] - Dates['start']).dt.days
    Dates["caseduration_seconds"] = (Dates['stop'] - Dates['start']).dt.seconds
    Dates.rename(columns={"id":"caseid"}, inplace=True)
    
    CaseData = pd.merge(left=CaseData,right=Dates, on="caseid")
    if verbose == True:
        print("done")
    #print("casedata:")
    #print(CaseData)
    return CaseData


def GetCaseStats(df, padded_df, CaseData, y_t, y_a, y, prefixwindow=0, dateformat="%Y-%m-%d %H:%M:%S", drop_last_ev=True, verbose=False):
    """  
    A table with one row per sequence (prefix) of padded_df, in the same order, which links the observations of the
    training data to their case and event: the identifier SEQID, the case and its statistics, the number of observed
    events (prefix_number), the start of the last observed event (prefix_date) and the target values y, y_t and y_a
    """
    import pandas as pd
    
    #one row per sequence, in the order of padded_df
    output = padded_df.drop_duplicates("SEQID")[["SEQID"]].reset_index(drop=True)
    
    #get case number and prefix number (the number of the last observed event) from the SEQID
    parts = output["SEQID"].str.split("_", expand=True)
    output["caseid"] = parts[0].astype(int)
    output["prefix_number"] = parts[1].astype(int)
    
    #case stats for every sequence
    output = pd.merge(left=output, right=CaseData, on="caseid", how="left")
    
    #Get event-specific data: the start of the last observed event of the prefix
    events = df[["id","event_number","time_parsed"]].rename(columns={"id":"caseid", "event_number":"prefix_number"}).copy()
    events["prefix_date"] = events["time_parsed"].dt.strftime(dateformat)
    output = pd.merge(left=output, right=events[["caseid","prefix_number","prefix_date"]], on=["caseid","prefix_number"], how="left")
    
    #the number of observations of the case: every event, or every event but the last
    if drop_last_ev:
        output["prefixes"] = output["num_events"]-1
    else:
        output["prefixes"] = output["num_events"]
    output["prefixwindow"] = prefixwindow
    
    #select what is interesting:
    output = output[["SEQID","caseid","log_caseid","num_events",
                     "prefix_number","prefixes","prefixwindow","prefix_date",
                     "distinct_events","caseduration_days"]]
    
    # step 2: Match the new table with target variables
    output["y"] = y.tolist()
    output["y_t"] = y_t.tolist()
    output = pd.concat([output.reset_index(drop=True), 
                        y_a.reset_index(drop=True).drop("caseid",axis=1)], axis=1)
    return output

def SplitAndReshape(df, y_a, y_t, y, split_criterion, prefixlength, verbose=False):
    import pandas as pd
    import numpy as np
    padded_df = df
    
    #prepare for join
    x = padded_df.reset_index(drop=True)
    x["caseid"] = x["caseid"].astype('int')
    
    y_a = y_a.reset_index(drop=True)
    y_a["caseid"] = y_a["caseid"].astype('int')
    
    y_t = pd.concat([y_a["caseid"], y_t], axis=1)
    y_t["caseid"] = y_t["caseid"].astype('int')
    y_t = y_t.reset_index(drop=True)
    
    y = pd.concat([y_a["caseid"], y], axis=1)
    y["caseid"] = y["caseid"].astype('int')
    y = y.reset_index(drop=True)
    
    
    split = split_criterion.reset_index(drop=True)
    split["caseid"] = split["caseid"].astype('int')
    
    ####################################################
    #Add the splitting cplumn (true = to trainset):
    #print("merging..")

    #print(len(x))
    X = pd.merge(left=x.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    #print(len(X))
    
    #print(len(y_a))
    y_a = pd.merge(left=y_a.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    #print(len(y_a))
    
    #print(len(y_t))
    y_t = pd.merge(left=y_t.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    #print(len(y_t))
    
    #print(len(y))
    y = pd.merge(left=y.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    #print(len(y))
    ####################################################
    #Subset based on the date divider, made in beginning:
    X_train = X.loc[X["trainset"]==True]
    X_test = X.loc[X["trainset"]==False]
    
    y_a_train = y_a.loc[y_a["trainset"]==True]
    y_a_test = y_a.loc[y_a["trainset"]==False]
    
    y_t_train = y_t.loc[y_t["trainset"]==True]
    y_t_test = y_t.loc[y_t["trainset"]==False]
    
    y_train = y.loc[y["trainset"]==True]
    y_test = y.loc[y["trainset"]==False]
    
    #Drop system variables
    X_train = X_train.drop(["caseid","SEQID","trainset"],axis=1)
    X_test = X_test.drop(["caseid","SEQID","trainset"],axis=1)
    
    y_a_train = y_a_train.drop(["caseid","trainset","event_number"], axis=1) #removing the ,"event_number" again
    y_a_test = y_a_test.drop(["caseid","trainset","event_number"], axis=1)
    
    y_t_train = y_t_train.drop(["caseid","trainset"], axis=1)
    y_t_test = y_t_test.drop(["caseid","trainset"], axis=1)
    
    y_train = y_train.drop(["caseid","trainset"], axis=1)
    y_test = y_test.drop(["caseid","trainset"], axis=1)
        
    ############################# 
    X_train = X_train.values
    X_test = X_test.values

    y_t_train = y_t_train.values
    y_t_test = y_t_test.values

    y_a_train = y_a_train.values
    y_a_test = y_a_test.values

    y_train = y_train.values
    y_test = y_test.values

    #Reshape:
    if verbose == True:
        print("reshaping..")
    
    #time, n, k
    timesteps = prefixlength
    observations = y_train.shape[0] #int(X.shape[0]/prefixlength)
    k = X_train.shape[1]
    
    #Reshape the data
    X_train = X_train.reshape(observations, timesteps, k)
    X_test = X_test.reshape(y_test.shape[0], timesteps, X_test.shape[1])
    if verbose == True:
        print("Trainset size (with prefixes of ",prefixlength,"):",y_train.shape[0])
        print("Testset size (with prefixes of ",prefixlength,"):",y_test.shape[0])
        print("==========================================")
        #Check the shapes
        print("X: observations, timesteps, vars")
        print(X_train.shape)
        print("y_train: observations, labels")
        print(y_train.shape)
        print("y_t_train: observations, labels")
        print(y_t_train.shape)
        print("y_a_train: observations, labels")
        print(y_a_train.shape)
    
    return X_train, X_test, y_t_train, y_t_test, y_a_train, y_a_test, y_train, y_test

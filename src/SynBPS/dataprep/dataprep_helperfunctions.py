# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:43:27 2019

@author: mikeriess
"""
import pandas as pd
import numpy as np


def InitialFormatting(df, maxcases, dateformat, verbose=False):
    import pandas as pd
    
    #Work on a subset:
    casestoload = df["id"].unique().tolist()[0:maxcases]
    df = df.loc[df["id"].isin(casestoload)]
    
    # find cases to drop due to length
    print("Cases before dropping len=1:",len(casestoload),"cases",len(df),"rows")
    
    # Make function to apply to groups
    def func(sub):
        
        out = None
        
        keepid = min(sub.id)
        
        if len(sub) > 1:
            out = keepid
        
        return out
    
    # Make list of cases above length 1
    df_grp = df.groupby('id').apply(func)
    
    #Remove NaNs from the list
    keepers = df_grp.values
    keepers = [i for i in keepers if i] 

    # Drop cases with only one event:
    df = df.loc[df["id"].isin(keepers)]
    
    print("Cases after dropping len=1:",len(keepers),"cases",len(df),"rows")
  
    #Sort the dataframe by time aftewards
    df['parsed_date'] = pd.to_datetime(df.time, format = dateformat, exact = True)
    
    ##########################################################################
    print("Sorting by id, date (chronological order)")
    #generate new ID column:
    df = df.assign(id=(df['id']).astype('category').cat.codes)
    
    df["id"] = df.id.astype('int32')
    
    # Ensure ID starts at 1
    if min(df.id) == 0:
        df.id = df.id +1
    
    # Sort the DF baed on caseid, and the date of the event
    df = df.sort_values(['id',"parsed_date"], ascending=[True, True]) 
    
    df = df.drop("parsed_date",axis=1)
    return df


def GetFileInfo(df):
    print("Number of cases in log:",len(df["id"].unique()))
    import numpy as np
    import pandas as pd
    
    #Get the maximal trace length, for determining prefix length
    max_length = np.max(df['id'].value_counts())
    print("longest trace is:",max_length)
    
    #Look at the time format:
    print("Time format:",df["time"].loc[0])
    print("Std. format: %Y-%m-%d %H:%M:%S")
    
    print(df.head())
    return max_length




def MakeSplitCriterion(df, trainsize=0.8, mode="event"):
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
        print("=======================================")
        print("Dropping cases that have events in both train + testsets:")
        print("=======================================")
        print("Cases before dropping:",len(validation["trainset"]))
        validation["keep"] = validation["trainset"] == 1
        validation = validation.loc[validation["keep"]==True]
        print("Cases after dropping:",len(validation["trainset"]))
        
        #list of caseids to keep
        ids_keep = validation["caseid"]
        
        #drop those thet should not be kept
        print("Total events before:",len(split_criterion))
        split_criterion = split_criterion.loc[split_criterion["caseid"].isin(ids_keep)]
        print("Total events after:",len(split_criterion))
        split_criterion = split_criterion.drop_duplicates(subset="caseid",keep="first")
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
    #Make copy of df
    data = df
    
    #Subset only relevant variables
    df = df[["id","time","event"]+category_cols+numeric_cols]
    
    
    import time as tm
    from datetime import datetime
    import pandas as pd
    import time
        
    # Make new case ids: ##############################
    cases = data["id"].unique().tolist()
    newcaseids = list(range(0,len(cases)))
    
    dictdf = pd.DataFrame([cases,newcaseids]).T
    dictdf.columns =["id","newid"]
    
    newdata = pd.merge(left=data,right=dictdf,on="id")
    newdata.rename(columns={'id':'dropme',
                            'newid':'id'}, 
                            inplace=False).drop("dropme",axis=1)
    
    # List all cases by their new id:
    cases = data["id"].unique().tolist()
    
    # Make new event ids: ##############################
    evids = []
    for i in cases:
        subset = data.loc[data["id"] == i]
        evids = evids + list(range(0,len(subset)))
    
    evids = [x+1 for x in evids] # + 1 ###################################################### added +1
    
    #set the new eventids
    data["eventid"] = evids 
    
    #make a counter to keep status
    num_cases = len(cases)
    
    # Generate features case by case
    for i in cases:
        #iteration = iteration +1
        if verbose==True:  
            print("case:",i, "of",num_cases)
    
        #Look only at one caseid at a time
        subset = data.loc[data["id"] == i]
        subset.index = subset.eventid
        
        """
        #######################################################################
        PREFIX:
        #######################################################################
        """
        
        index1 = 0
        
        #determine whether to start in the beginning or end of trace        
        if window_position == "last_k":
        
            #if trace is smaller than desired prefix, just pick the full trace
            if max_prefix_length > len(subset):
                start = 1 #0
                stop = len(subset) - index1 # 
            
            #If the the max prefix len is smaller than the actual trace, 
            #take the K last events (sliding window approach)
            if max_prefix_length < len(subset):
                start = len(subset) - max_prefix_length
                stop = len(subset) - index1
    
            #If max prefix is identical to trace len, start from one        
            if max_prefix_length == len(subset):
                start = 1 #0
                stop = len(subset) - index1
                
                
                
        if window_position == "first_k":
        
            #if trace is smaller than desired prefix len, just pick the full trace
            if max_prefix_length > len(subset):
                start = 1 #0
                stop = len(subset) - index1 # 
            
            #If the the max prefix len is smaller than the actual trace, 
            #take the K FIRST events (sliding window approach)
            if max_prefix_length < len(subset):
                start = 1
                stop = max_prefix_length - index1
    
            #If max prefix is identical to trace len, start from one        
            if max_prefix_length == len(subset):
                start = 1 #0
                stop = len(subset) - index1
        if verbose==True:
            print("start",start,"stop",stop)
        
        #Prefix capability: Subset k last events from trace
        subset = subset.loc[start:stop]
        
        #Make sure the data to be dummified also follows prefix convention
        if i == 1:
            datasub = subset
            if verbose==True:
                print("len subset:",len(subset))
                print("len datasub:",len(datasub))

        if i > 1:
            datasub = pd.concat([datasub, subset],axis=0)
            
        """
        #######################################################################
        PREFIX:
        #######################################################################
        """
        
        #Get list of events
        eventlist = subset.eventid.tolist()
        
        #store the case id
        caseid = str(i)
        
        
        #for every event:
        for event in eventlist:
            #print(event)
            
            #Generate an eventID
            event_number = event+1
            
            #get the event for later reference
            event_activity = subset["event"].loc[event]
            
            #For the tax-approach, get next event
            if event != stop: #len(subset)-1: #if its the last event
                next_activity = subset["event"].loc[event+1]
            
            if event == stop: #len(subset)-1: #if its the last event
                next_activity = "END"
                
                    
            """
            #######################################################################
            Beginning of time features:
            """
            
            #first event
            starttime = datetime.fromtimestamp(tm.mktime(tm.strptime(subset["time"].loc[start], dateformat)))
         
            #time in secs since midnight
            t = tm.strptime(subset["time"].loc[event], dateformat) #Time now
            midnight = datetime.fromtimestamp(tm.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0) #Midnight
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
    
            #monday = 1
            dayofweek = datetime.fromtimestamp(tm.mktime(t)).weekday()+1 
            
            #hour of day
            hour = datetime.fromtimestamp(tm.mktime(t)).hour
    
            #Time since start in seconds
            timesincestart = (datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()
    
            #Time since last event in seconds
            if event ==start:
                t_last = tm.strptime(subset["time"].loc[event], dateformat) #Time last event: now
                timesincelastev = 0
            if event !=start:
                t_last = tm.strptime(subset["time"].loc[event-1], dateformat) #Time last event
                timesincelastev = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t_last))).total_seconds()
          
            #Time until finishtime
            t_finish = tm.strptime(subset["time"].loc[stop], dateformat) #Time last event
            timetofinish = (datetime.fromtimestamp(time.mktime(t_finish)) - datetime.fromtimestamp(time.mktime(t))).total_seconds()
    
            
            #Time to next event:
            if event == stop:#len(subset)-1: #if its the last event
                t_nextev = tm.strptime(subset["time"].loc[event], dateformat) #Time last event: now
                timetonextev = 0
                
            if event != stop:#len(subset)-1: #if not last event
                t_nextev = tm.strptime(subset["time"].loc[event+1], dateformat) #Time last event
                timetonextev = (datetime.fromtimestamp(time.mktime(t_nextev))-datetime.fromtimestamp(time.mktime(t))).total_seconds()
    
            """
            #######################################################################
            End of time features for each event
            """
                    
            #Make a marker for dropping last step where remaining time (y) = 0
            drop = 0
            #Mark if it is the last activity:
            if event == stop:#len(subset)-1:
                drop = 1
                
               
            #Actual:
            results = [caseid,
                       event_number,
                       event_activity, #event
                       timesincemidnight,
                       dayofweek,
                       hour,
                       timesincestart,
                       timesincelastev,
                       timetofinish,
                       timetonextev,
                       next_activity,
                       drop]
            
            if i == 1 and event == start: #First time
                out = pd.DataFrame(results).T
                
            else:
                res_i = pd.DataFrame(results).T
                #out = out.append(res_i)
                #new try (21/12/23)
                out = pd.concat([out, res_i],axis=0)
            
    #Rename all static vars
    cols = ['caseid',
            'event_number',
            'event_activity',
            'timesincemidnight',
            'dayofweek',
            'hourofday',
            'timesincestart',
            'timesincelastev',
            'y_timetofinish',
            'y_timetonextev',
            'next_activity',
            'drop']
    out.columns = cols
      
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
            print("\nDummification of",dummylist)
    
            Dummies = pd.get_dummies(datasub[dummylist])
            Dummies = Dummies.reset_index(drop=True)
            out = out.reset_index(drop=True)
            dummycols = Dummies.columns.tolist()
            out = pd.concat([out,Dummies],axis=1)
        
        if len(numeric_cols) > 0:
            #add numerical features:
            print("Adding numerical features:",numeric_cols)
            numerics = datasub[numeric_cols]
            numerics = numerics.reset_index(drop=True)
            numerics = numerics.add_prefix('num_')
            out = out.reset_index(drop=True)
            out = pd.concat([out,numerics],axis=1)
    
    
    if dummify_time_features == True:
        features = ["dayofweek","hourofday"]
        print("Dummification of time features",features)
        sysdummies = out[features]
        sysdummies = pd.get_dummies(sysdummies,prefix="t_")
        sysdummies = sysdummies.reset_index(drop=True)
        out = out.drop(features,axis=1)
        out = out.reset_index(drop=True)
        out = pd.concat([out, sysdummies],axis=1)
    
    #Remove last event in each case
    if droplastev==True:
        print("\ndropping last event from each case")
        print("before:",len(out))
        out = out.loc[out["drop"] != 1]
        out = out.drop("drop",axis=1)
        print("after:",len(out))
        print("data in X is the",max_prefix_length,"last events, excluding the final event")
    
    #####################################
    # Separate outputs
    #####################################
    
    print("\ndropping vars:")
    
    # Generate the next_activity target vector
    y_a = out[y_a_new_varnames]
    y_a["caseid"] = out["caseid"]
    y_a["event_number"] = out["event_number"]
    
    
    if drop_end_target==True:
        """ NOT SURE IF THIS SHOULD BE DONE FOR THE TAX MODEL
            as this would signal the end of a trace..
        """
        dropme = ['y_a_t1_1']#before it was: ["y_a_t1_END"]
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



def PadInputs(caseids, df, max_prefix_len=3, standardize=True, verbose=False):    
    import pandas as pd
    import numpy as np
    import time
    #copy the dataframe
    res = df
    maxlen = max_prefix_len
    
    #Make an empty placeholder of the dataset
    dataset = pd.DataFrame(df.loc[0]).T
    dataset["SEQID"] = 0 #placeholder
    dataset = dataset.drop(0, axis=0)
    
    #Make a counter
    count = 0
    allcases = len(caseids)
    timeend = time.time()
    
    #loop through the cases
    for i in caseids:
        count = count +1
        
        #Get only events from case i
        subset = res.loc[res["caseid"] == str(i)]
        
        events = subset["event_number"].unique().tolist() #event
        cols = subset.columns    
        
        #time display
        timestart = time.time()
        timetaken=np.round((timestart-timeend),decimals=3)
        timeleft=np.round(((allcases-count)*(timetaken*60))/60,decimals=2)
        
        if verbose==True:
            print("Case:",count,"of ",allcases," events:",len(events),
                  "-",timetaken,"s. per case, est.",timeleft,"min. left")

        #for row in subset.itertuples():
        for j in events:
            #print(row)
            #j = int(row.event_number)            
            """row = subset.loc[subset["event_number"] == j]
            
            #if it is touple form, try 2nd colum/datapoint
            j = int(row.event_number)"""

            
            ##################
            # Changing this to the event number, rather than type
            #j = int(row.eventno)
            ##################
            
            #Get current timestep, and all earlier timesteps
            EV = subset.loc[subset["event_number"] < j+1]
            
            #Figure out how many rows to pad
            rowstoadd = maxlen - len(EV)
            
            ### Padding: pre-event-padding ###
            zeros = np.zeros((rowstoadd, EV.shape[1]))
            zeros = pd.DataFrame(zeros, columns=cols)
            
            #Add the zeros before the actual data
            EV = pd.concat([zeros, EV], ignore_index=True, axis=0)
            
            #Set an ID for the sequence
            EV["SEQID"] = str(i)+"_"+str(j)
            EV["caseid"] = str(i)
            
            #Add the sequence to the dataset
            #dataset = dataset.append(EV)
            dataset = pd.concat([dataset,EV], axis=0)
        timeend = time.time()

    
    print("\n\nOutput length:",len(dataset))
    return dataset



def CaseData(df):
    print("Generating case data")
    # step 0: Get case aggregate stats:
    import pandas as pd
    CaseData = pd.DataFrame(df['id'].value_counts())
    CaseData.rename(columns={"count":"num_events"}, inplace=True)
    CaseData["caseid"] = CaseData.index
    CaseData.sort_values('caseid', inplace=True)
    
    distinct_events = df.groupby('id').event.nunique()
    CaseData["distinct_events"] = distinct_events
    
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
    print("done")


    #print("casedata:")
    #print(CaseData)
    return CaseData


def GetCaseStats(df, padded_df, CaseData, y_t, y_a, y, prefixwindow=0, dateformat="%Y-%m-%d %H:%M:%S", drop_last_ev=True, verbose=False):
    """  
    What we want is a table that can be linked with the 
    observations in the training data. 
    
    This table has to be ordered, and then permuted the 
    exact same way as the X/y output of this pipeline.
    
    Table:
        - SEQID (level)
            - Input table (prefixes):
                - Event number (same as number of events in curr. SEQID)
                - number of events in parent case (trace length)
                - (other interesting attributes could be added here)
                
            - Target values
                - y
                - y_t
                - y_a
    
        - Dataset level (aggregated):
            - number of prefixes in dataset
            - number of unique cases
            - number of events 
            - average number of events per case
            - (other stats from the survey paper)
    """

    #Get all SEQIDs
    SEQIDS = padded_df["SEQID"].unique().tolist()
    
    allseqs = len(SEQIDS)
    #logic to build output table
    counter = 0
    print("Making casestats..")
    for i in SEQIDS:
        if verbose == True:
            print("Making casestats for SEQ:",counter,"of",allseqs)
        counter = counter +1    
        prefix = padded_df.loc[padded_df["SEQID"]==i]
                
        SEQID = i

        #get case number and prefix number
        caseno = int(prefix["SEQID"].loc[0].split("_")[0])
        eventno = int(prefix["SEQID"].loc[0].split("_")[1])
        
        if drop_last_ev == False:
            #Add 1 to the index in the case all events are present
            eventno = eventno - 1
        
        #Get event-specific data:
        case = df.loc[df["id"]==caseno]
        event = case.loc[case["event_number"]==eventno]
        
        eventtime = event["time_parsed"].dt.strftime(dateformat).tolist()[0]
            
        #get case stats for current seuqnce/prefix:
        casestats = CaseData.loc[CaseData["caseid"]==caseno]
        
        if verbose == True:
            print("casestats")
            print(casestats)
        
        casestats["prefix_date"] = eventtime        
        casestats["prefixes"] = casestats["num_events"]-1
        casestats["prefixwindow"] = prefixwindow
        casestats["prefix_number"] = eventno
        casestats["SEQID"] = SEQID
        
        #select what is interesting:
        casestats = casestats[["SEQID","caseid","num_events",
                               "prefix_number","prefixes","prefixwindow","prefix_date",
                               "distinct_events","caseduration_days"]]

        if counter == 1:
            output = casestats
        if counter > 1:
            output = pd.concat([output,casestats],axis=0)
    
    # step 2: Match the new table with target variables
    #"""    
    print("Done.")
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
    print("reshaping..")
    
    #time, n, k
    timesteps = prefixlength
    observations = y_train.shape[0] #int(X.shape[0]/prefixlength)
    k = X_train.shape[1]
    
    #Reshape the data
    X_train = X_train.reshape(observations, timesteps, k)
    X_test = X_test.reshape(y_test.shape[0], timesteps, X_test.shape[1])
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

"""
GENR the experiments:
"""

def fullfact_corrected(levels):
    import numpy as np
    import pandas as pd
    """
    Create a general full-factorial design
    
    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.
    
    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor
    
    Example
    -------
    ::
    
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
               
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H

def construct_df(x,r):
    import numpy as np
    import pandas as pd
    df=pd.DataFrame(data=x,dtype='float32')
    for i in df.index:
        for j in range(len(list(df.iloc[i]))):
            df.iloc[i][j]=r[j][int(df.iloc[i][j])]
    return df

def build_full_fact(factor_level_ranges):
    import numpy as np
    import pandas as pd
    """
    Builds a full factorial design dataframe from a dictionary of factor/level ranges
    Example of the process variable dictionary:
    {'Pressure':[50,60,70],'Temperature':[290, 320, 350],'Flow rate':[0.9,1.0]}
    """
    
    factor_lvl_count=[]
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lvl_count.append(len(factor_level_ranges[key]))
        factor_lists.append(factor_level_ranges[key])
    
    x = fullfact_corrected(factor_lvl_count)
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df
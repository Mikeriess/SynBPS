#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def run_experiments(dataprep_function, training_function, eval_function, store_eventlogs=False, output_dir="data/", out_file="results.csv", design_table="design_table.csv", verbose=False):
    """
    Function for running experiments
    """
    
    # Load necessary libraries
    import pandas as pd
    import numpy as np
    import time
    from tqdm import tqdm


    # load the design table as df
    df = pd.read_csv(output_dir+design_table)
    
    # Placeholder for the results
    results = []

    # Iterate over each run in the design table df
    for run in tqdm(df.index):
        
        """
        Retrieving settings for experiment i
        """
        curr_settings = df.loc[run]
        curr_settings["idx"] = run
        
        """
        If experiment is not previously performed
        """
        if curr_settings.Done == 0:
            if verbose==True:
                print("Run:",run)
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

            input_data = dataprep_function(log)

            """
            Train a model
            """
            
            ### Custom training function
            inference_test = training_function(input_data)
            
            if store_eventlogs==True:
                # store inference table
                inference_test.to_csv(output_dir+"inference_test_"+str(run)+".csv", index=False)

            """
            Evaluate the model
            """

            ### Custom evaluation function
            metrics = eval_function(inference_test)
            
            """
            Store the results
            """
            
            # Mark as done in the design table
            df.loc[run,"Done"] = 1
            df.to_csv(output_dir + design_table, index=False)  
            
            ### Store metrics to the in curr_settings dictionary which becomes the result table
            ### Prefixing column names is ideal for later analysis
            
            curr_settings["RESULT_num_events"] = len(log)
            
            # add evaluation metrics
            metrics = pd.DataFrame(metrics, index=[run])
            curr_settings = curr_settings.to_dict()
            curr_settings = pd.DataFrame(curr_settings, index=[run])
            res_i = pd.concat([curr_settings, metrics], axis=1)
            
            # Store the settings of run i
            results.append(res_i)
    
            #store results
            experiments = pd.concat(results)
            experiments.to_csv(output_dir+out_file, index=False)
                    
    return experiments

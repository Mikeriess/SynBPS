#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def run_experiments(training_function, eval_function, store_eventlogs=False, output_dir="data/", out_file="results.csv", design_table="design_table.csv", verbose=False):
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
            for traceid in log.caseid.unique():
                trace = log.loc[log.caseid == traceid]
                
                #tracelen
                tracelen = len(trace)
                tracelengths.append(tracelen)
                
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
    
            """
            Prepare data for modelling (memory here refers to RAM)
            """

            # Prefix-log format:
            from SynBPS.dataprep.prepare import prefix_data
            input_data = prefix_data(log, verbose=False)

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

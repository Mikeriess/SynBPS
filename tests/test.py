
# Suppress warnings in this notebook
import warnings
warnings.filterwarnings('ignore')

# Load necessary libraries
import pandas as pd
import numpy as np
import time

# Define settings
run_settings = {# level of entropy: min, medium and/or max
                "process_entropy":["min_entropy","med_entropy","max_entropy"],
                
                # number of traces/cases in the event-log
                "number_of_traces":[100],

                # number of activity types
                "statespace_size":[5,10], 

                # first or higher-order markov chain to represent the transitions
                "process_type":["memoryless","memory"], 
                
                # order of HOMC - only specify this when using process with memory
                "process_memory":[2,4],
                
                # number of transitions - only used for medium entropy (should be higher than 2 and < statespace size)
                "med_ent_n_transitions":[3, 5],
                                
                # lambda parameter of inter-arrival times
                "inter_arrival_time":[1.5],
                
                # lambda parameter of process noise
                "process_stability_scale":[0.1],
                
                # probability of agent being available
                "resource_availability_p":[0.5],

                # number of agents in the process
                "resource_availability_n":[3],

                # waiting time in full days, when no agent is available
                "resource_availability_m":[0.041],
                
                # variation between activity durations
                "activity_duration_lambda_range":[1, 5],
                
                # business hours definition: when can cases be processed? ()
                "Deterministic_offset_W":["weekdays", "all-week"],

                # time-unit for a full week: days = 7, hrs = 24*7, etc.
                "Deterministic_offset_u":[7],
                
                # train machine learning model from simulated event-logs 
                "model_pipeline":[False],

                # offset for the timestamps used (1970 time after 1970)
                "datetime_offset":[35],
                
                # number of repetitions of the experiments: duplicates the experiment table (2 times here)
                "num_replications":list(range(0, 2))
               }


# import the make_design_table function to generate a full factorial experimental design table
from SynBPS.simulation.DoE import make_design_table
df = make_design_table(run_settings, file="data/design_table.csv")

# inspect the resulting design table
df

# Placeholder for the results
results = []

# Iterate over each run in the design table df
for run in df.index:
    
    """
    Retrieving settings for experiment i
    """
    curr_settings = df.loc[run]
    curr_settings["run"] = run
    
    """
    If experiment is not previously performed
    """
    if curr_settings.Done == 0:
        print("Run:",run)
        start_time = time.time()

        # generate the log
        from SynBPS.simulation.simulation_pipeline import generate_eventlog

        log = generate_eventlog(curr_settings=curr_settings, output_dir="data/")
        
        #log.to_csv("results/"+str(run)+"log.csv",index=False)

        # store metrics from simulated log
        curr_settings["simuation_time_sec"] = time.time() - start_time
        curr_settings["num_traces"] = len(log.caseid.unique())
        curr_settings["num_events"] = len(log)

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
        Run the machine learning pipeline from here
        """

        if curr_settings["model_pipeline"] == True:
            """
            Prepare data for modelling (memory here refers to RAM)
            """
            from SynBPS.dataprep.prepare import prefix_data
            input_data = prefix_data(log, verbose=False)

            """
            Train a model
            """
            # X: 
            input_data["x_train"]
            input_data["x_test"]

            # Y:
            input_data["y_train"]
            input_data["y_test"]
            
            
            
            ### <<<<<< Insert training functions here >>>>>>>
            

            """
            Evaluate the model
            """

            ### <<<<<< Insert evaluation functions here >>>>>>>
            
            """
            Store the results
            """
            
            ### Store metrics to the in curr_settings dictionary which becomes the result table
            ### Prefixing column names is ideal for later analysis

            curr_settings["RESULT_num_events"] = len(log)
        
        #Mark as done in the design table
        df.loc[run,"Done"] = 1
        df.to_csv("data/design_table.csv",index=False)  
        
        #Store the settings of run i
        results.append(curr_settings)
        
                
#store results
experiments = pd.DataFrame(results)
experiments.to_csv("data/experiments.csv",index=False)  
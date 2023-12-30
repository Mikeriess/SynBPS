.. example:


Usage
===================
SynBPS is designed to be used in the following manner:

* Generate experimental design table (table of all settings to be simulated)
* Specify Train() and Eval() functions (to be used in each experiment)
* Run experiments (using your approach)
* Analyze results

Example use-case
------------------

1. Generate design table
Here the settings for the experiments can be modified in the dictionary called run_settings. Refer to the paper for more details on of each of the parameters.

.. code-block:: python

    run_settings = {# level of entropy: min, medium and/or max
                "process_entropy":["min_entropy","med_entropy","max_entropy"],
                
                # number of traces/cases in the event-log
                "number_of_traces":[1000],

                # number of activity types
                "statespace_size":[5, 10], 

                # first or higher-order markov chain to represent the transitions
                "process_type":["memoryless","memory"], 
                
                # order of HOMC - only specify this when using process with memory
                "process_memory":[2, 4],
                
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
                
                # training data format (See Verenich et al., 2019): 
                # True - use first event to predict total cycle-time. 
                # False - use Prefix-log format / each event to predict remaining cycle time.
                "first_state_model":[True],

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

2. Specify Train() and Test() functions
Before running the experiments, you need to define model training and evaluation functions.

In this example we train a first state model, which is a model using only the first observed event (state) to predict to total cycle-time. The default data preparation will result in a prefix-log, which can be used to predict remaining cycle-time from every observed event in the trace.

Input for the **training_function** is a dictionary named **input_data**, which contain the following:
- x_train
- x_test
- y_train
- y_test

Output is an **inference table** containing predictions and actual target values for the test data. This table is used for analysis of the results. The **eval_function** also uses this table to calculate aggregated metrics.

.. code-block:: python

    def training_function(input_data):
        print("training")
        
        """
        Example model: Lasso regression
        This is just an example of how to define your model in this framework.
        Using this model on this data format is not advised as we break i.i.d. assumptions.
        """

        #retrieve model class from sklearn
        from sklearn import linear_model
        reg = linear_model.Lasso(alpha=0.1)

        #reshape training data for this type of model 
        #(from: N x t x k, to: N x (t x k))
        #num_obs = input_data["x_train"].shape[0]
        from numpy import prod
        flattened_dim = prod(input_data["x_train"].shape[1:])

        #train the regression model
        reg.fit(input_data["x_train"].reshape((input_data["x_train"].shape[0], flattened_dim)), input_data["y_train"])

        #predict on the test data
        y_pred = reg.predict(input_data["x_test"].reshape((input_data["x_test"].shape[0], flattened_dim)))

        #get the inference table (used for analysis of the final results)
        inference = input_data["Inference_test"]
        
        #add predictions to the inference table
        inference["y_pred"] = y_pred
        return inference

    def eval_function(inference):
        print("evaluation")

        """
        Example evaluation: Aggregated scores
        The inference table also enable the ability to make trace or prefix-level evaluations using its id variables
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

        y = inference["y"]
        y_pred = inference["y_pred"]

        MSE = mean_squared_error(y, y_pred)
        MAE = mean_absolute_error(y, y_pred)
        R2 = r2_score(y, y_pred)
        EVAR = explained_variance_score(y, y_pred)

        # the resulting metrics should be stored in a dictionary and be scalars only
        # adding prefixes to column name (key) is suggested when logging many metrics
        metrics = {"TEST_MSE":MSE,
                "TEST_MAE":MAE,
                "TEST_R2":R2,
                "TEST_EVAR":EVAR}
        print(metrics)
        return metrics

3. Run experiments

.. code-block:: python

    # Run experiments
    from SynBPS.simulation.simulation_pipeline import run_experiments
    results = run_experiments(training_function, 
                            eval_function, 
                            output_dir="data/",
                            out_file="results.csv", 
                            design_table="design_table.csv")

4. Analyze results

.. code-block:: python

    # This is still a work in progress, however the results will be placed in output_dir
    # and can be analyzed using pandas or other tools.
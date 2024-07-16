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

1. **Generate design table**
Here the settings for the experiments can be modified in the dictionary called run_settings. Refer to the paper for more details on of each of the parameters.

.. code-block:: python

    # dictionary consisting of all desired settings for each factor
    run_settings = {
                # level of entropy: min, medium and/or max
                "process_entropy":["min_entropy"], #,"med_entropy","max_entropy"
                
                # number of traces/cases in the event-log
                "number_of_traces":[500],

                # number of activity types
                "statespace_size":[5], 

                # first or higher-order markov chain to represent the transitions
                "process_type":["memoryless"], 
                
                # order of HOMC - only specify this when using process with memory
                "process_memory":[2],
                
                # number of transitions - only used for medium entropy (should be higher than 2 and < statespace size)
                "med_ent_n_transitions":[1,2,3,4,5],
                                
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
                "activity_duration_lambda_range":[1],
                
                # business hours definition: when can cases be processed? ()
                "Deterministic_offset_W":["weekdays"],

                # time-unit for a full week: days = 7, hrs = 24*7, etc.
                "Deterministic_offset_u":[7],
                
                # training data format (See Verenich et al., 2019): 
                # True - use first event to predict total cycle-time. 
                # False - use Prefix-log format / each event to predict remaining cycle time.
                "first_state_model":[False],

                # offset for the timestamps used (years after 1970)
                "datetime_offset":[45],
                
                # number of repetitions of the experiments: duplicates the experiment table (5 times here)
                "num_replications":list(range(0, 5))
            }

    # import the make_design_table function to generate a full factorial experimental design table
    from SynBPS.design.DoE import make_design_table
    df = make_design_table(run_settings)

    # give each run its own seed value such that results are different across runs
    df["seed_value"] = range(0,len(df))

    # store the design table
    df.to_csv("data/design_table.csv", index=False)

    # inspect the resulting design table
    df.head()

2. **Specify Train() and Test() functions**
Before running the experiments, you need to define model training and evaluation functions.

In this example we train a first state model, which is a model using only the first observed event (state) to predict to total cycle-time. The default data preparation will result in a prefix-log, which can be used to predict remaining cycle-time from every observed event in the trace.

Input for the **training_function** is a dictionary named **input_data**, which contain the following:
* x_train
* x_test
* y_train
* y_test

The default behavior of the data preparation is a temporal split with 70 percent of the data in train and 30 in the test set. Feel free to modify the data preparation steps in dataprep/prepare.py

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

Output is an **inference table** containing predictions and actual target values for the test data. This table is used for analysis of the results. The **eval_function** also uses this table to calculate aggregated metrics.

.. code-block:: python

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

3. **Run experiments**
The experiments can be run using the **run_experiments** function, which takes the training function and evaluation function specified above as its first two arguments. Next, the output directory of the data created during the experiments needs to be specified (here we use **data/**), followed by the destination file to store the results, and the input design table created in step 1 of this guide. 
.. code-block:: python

    # function to run a set of experiments
    from SynBPS.simulation.simulation_pipeline import run_experiments

    # run experiments
    results = run_experiments(training_function, 
                            eval_function, 
                            output_dir="data/",
                            out_file="results.csv", 
                            design_table="design_table.csv")

4. **Analyze results**
Firstly we load the results table which contain aggregated metrics based on the individual runs. This can then be plotted and analyzed in any manner desired.

.. code-block:: python

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the results
    df = pd.read_csv("data/results.csv")

    # Create boxplot
    sns.boxplot(data=df, x='med_ent_n_transitions', y='TEST_R2')

    # Calculate medians and plot lines
    medians = df.groupby(['med_ent_n_transitions'])['TEST_R2'].median().values
    n = len(medians)
    sns.lineplot(x=range(n), y=medians, sort=False)

    # Set title and y-axis range
    plt.title('Boxplot with Median Lines')
    plt.ylim(0, 1)

    plt.show()

.. example:


Usage (high level)
===================
SynBPS is designed to be used in the following manner:

1. Generate design table (table of all settings to be simulated)
2. Specify Train() and Test() functions
3. Run experiments
4. Analyze results

Example use-case
------------------

1. Generate design table

.. code-block:: python

    # Generate a design table
    design_table = DesignTable()

    # Add all settings to the design table
    design_table.add_setting('learning_rate', [0.1, 0.01, 0.001])
    design_table.add_setting('batch_size', [32, 64, 128])
    design_table.add_setting('epochs', [10, 20, 30])

2. Specify Train() and Test() functions

.. code-block:: python

    # Define a Train() function
    def Train(setting):
        # Create a model with the given setting
        model = create_model(setting)

        # Train the model
        model.train()

        # Return the trained model
        return model

    # Define a Test() function
    def Test(model, setting):
        # Test the model with the given setting
        accuracy = model.test()

        # Return the accuracy
        return accuracy

3. Run experiments

.. code-block:: python

    # Run experiments
    results = design_table.run(Train, Test)

4. Analyze results

.. code-block:: python

    # Get the best setting
    best_setting = results.get_best_setting()

    # Get the best accuracy
    best_accuracy = results.get_best_accuracy()

    # Plot the results
    results.plot()
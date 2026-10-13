.. dataprep:


Preparing an event-log for a model
===================================
``prefix_data`` in ``SynBPS.dataprep.prepare`` turns an event-log into a prefix-log: one observation per observed event of every trace, with the events seen so far as a padded sequence, and the remaining time, the time to the next event and the next activity as targets. This is the input format of the sequence models used in predictive process monitoring. It can be used as the ``dataprep_function`` of :doc:`example` as it is, or as a starting point for your own preparation.

.. code-block:: python

    from SynBPS.dataprep.prepare import prefix_data

    data = prefix_data(log)

    data["x_train"].shape      # (observations, prefix length, features)
    data["y_train"].shape      # (observations, 1): remaining time in seconds
    data["Inference_test"]     # one row per observation of the test set

Steps
------
1. The columns ``caseid``, ``activity``, ``activity_no``, ``start_datetime`` and ``end_datetime`` of the log are kept, together with the columns named in ``category_cols`` and ``numeric_cols``. The timestamps are the start of each event.
2. Traces with a single event are dropped, as there is nothing to predict for them. The remaining cases are renumbered 1, 2, 3, ... in the order of the log. The original ``caseid`` is kept in the column ``log_caseid`` of the inference tables.
3. The prefix length is the number of events of the longest trace, so every event of every trace is kept.
4. The traces are split by time: the dates of the log are divided at ``trainsize`` (70 percent by default), and a trace belongs to the train set when its first event starts before the split date.
5. For every event, the features and targets below are computed from the events of its trace. The last event of every trace is dropped afterwards, as its remaining time is 0.
6. One observation (a prefix) is built per remaining event: the rows of the events up to and including it, with rows of zeros in front, so that every observation has the prefix length as its number of rows.
7. The numeric features are scaled when ``scaling`` is set, with the parameters of the train set.
8. The observations are split into train and test arrays, and an inference table is built in the same order.

Features
---------
The features of an event, in the order of the last axis of ``x_train``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Feature
     - Meaning
   * - ``event_number``
     - Number of the event in its trace, 1 for the first event
   * - ``timesincemidnight``
     - Seconds since midnight at the start of the event
   * - ``dayofweek``
     - Day of the week, 1 for Monday to 7 for Sunday. One-hot encoded (``t__1`` to ``t__7``) with ``dummify_time_features``
   * - ``hourofday``
     - Hour of the day, 0 to 23. One-hot encoded with ``dummify_time_features``
   * - ``timesincestart``
     - Seconds since the start of the first event of the trace
   * - ``timesincelastev``
     - Seconds since the start of the previous event, 0 for the first event
   * - ``ev_a_t0_1`` to ``ev_a_t0_d``
     - One-hot encoding of the activity of the event, numbered in alphabetical order of the activity names
   * - ``<column>_<value>``
     - One-hot encoding of every column in ``category_cols``, for example ``start_day_Monday``
   * - ``num_<column>``
     - Every column in ``numeric_cols`` as it is in the log, for example ``num_h_t``

The padding rows of an observation are 0 in every feature, also when the features are scaled. The features are not scaled by default: times are in seconds and can be large, so set ``scaling`` to ``"standardize"`` ((x - mean) / std) or ``"normalize"`` ((x - min) / (max - min)) when the model needs it. The parameters are computed from the train set only, applied to both sets, and returned in ``data["scaling"]`` per feature. One-hot features are never scaled.

Targets
--------
.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Array
     - Meaning
   * - ``y_train``, ``y_test``
     - Remaining time of the trace in seconds: from the start of the event to the start of the last event of the trace
   * - ``y_t_train``, ``y_t_test``
     - Time to the next event in seconds
   * - ``y_a_train``, ``y_a_test``
     - The next activity, one-hot encoded with one column per activity in alphabetical order (``y_a_t1_2`` to ``y_a_t1_d+1``). The column for the end of the trace is dropped, as the last event of every trace is not an observation

The targets are not scaled.

Inference tables
-----------------
``Inference_train`` and ``Inference_test`` have one row per observation, in the same order as the arrays, so a prediction can be linked back to its trace and event:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``SEQID``
     - ``<caseid>_<prefix_number>``, the identifier of the observation
   * - ``caseid``
     - The renumbered case, 1, 2, 3, ...
   * - ``log_caseid``
     - The ``caseid`` of the event-log, to join with the log
   * - ``num_events``
     - Number of events of the trace
   * - ``prefix_number``
     - Number of observed events in the observation, which is also the ``event_number`` of its last event
   * - ``prefixes``
     - Number of observations of the trace, ``num_events - 1``
   * - ``prefixwindow``
     - The prefix length
   * - ``prefix_date``
     - Start of the last observed event
   * - ``distinct_events``
     - Number of distinct activities of the trace
   * - ``caseduration_days``
     - Whole days from the first to the last event of the trace
   * - ``y``, ``y_t``, ``y_a_t1_...``
     - The targets of the observation
   * - ``event_number``
     - The ``event_number`` of the last observed event, the same as ``prefix_number``

Adding features
----------------
Every column of the event-log can be added as a feature of its event: categorical columns with ``category_cols`` (one-hot encoded) and numeric columns with ``numeric_cols``. For example, the weekday of the start of the event, the resource offset and the business-hours offset:

.. code-block:: python

    data = prefix_data(log, category_cols=["start_day"], numeric_cols=["h_t", "s_t"], scaling="standardize")

Note that most columns of a SynBPS log describe the event itself, such as ``v_t`` (the duration of the activity) or ``u_t``. Using them as features of the same event means that the model sees the outcome of the event it is at. Columns of earlier events are safe, which is what the sequence of a prefix provides.

Features which cannot be expressed as a column of the log need a custom ``dataprep_function``. The helper functions in ``SynBPS.dataprep.dataprep_helperfunctions`` can be reused for that: ``prefix_data`` is a short function which calls them in the order of the steps above.

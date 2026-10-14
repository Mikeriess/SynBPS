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
    data["cases"]              # one row per case of the log: train, test or dropped
    data["split_date"]         # the date which separates the train and the test set

Steps
------
1. The cases are assigned to the train set, the test set or neither, by the rules of `Train and test split`_ below. Only the cases of the train and test set are used from here on.
2. The columns ``caseid``, ``activity``, ``activity_no``, ``start_datetime`` and ``end_datetime`` of the log are kept, together with the columns named in ``category_cols`` and ``numeric_cols``. The timestamps are the start of each event.
3. The cases are renumbered 1, 2, 3, ... in the order of the log. The original ``caseid`` is kept in the column ``log_caseid`` of the inference tables and of ``cases``.
4. The prefix length is the number of events of the longest trace, so every event of every trace is kept.
5. For every event, the features and targets below are computed from the events of its trace. The last event of every trace is dropped afterwards, as its remaining time is 0, unless ``include_end`` is True.
6. One observation (a prefix) is built per remaining event: the rows of the events up to and including it, with rows of zeros in front, so that every observation has the prefix length as its number of rows.
7. The numeric features are scaled when ``scaling`` is set, with the parameters of the train set.
8. The observations are split into train and test arrays, and an inference table is built in the same order.

Train and test split
---------------------
The split is by time: the cases which start first are the train set, the cases which start later are the test set. The rules are applied in this order, with the parameters in brackets:

1. **Cases with a single event** are dropped (``drop_single_event_cases``, default True). Such a case has nothing to predict, unless ``include_end`` is True: then its one event is an observation whose next activity is END.
2. **The split date** is computed from ``trainsize`` (default 0.7):

   * with ``split_by="cases"`` (default), the cases are ordered by the start of their first event, and the split date is the start of the first case in the test set, such that a share of ``trainsize`` of the cases is in the train set. Cases which start at the same time cannot be separated by a date (with business hours, many cases start at the opening hour), so when the split falls in such a group, the split date is placed before or after the group, whichever gives a share closer to ``trainsize``;
   * with ``split_by="date"``, the split date is the start of the earliest event plus ``trainsize`` of the days between the earliest and the latest event, so ``trainsize`` is a share of the calendar span of the log and the share of cases depends on the arrivals. This was the only rule up to version 1.1.9.

   A date given as ``split_date`` (a timestamp or a string such as ``"2024-03-01"``) replaces the computed one.
3. **A case is in the train set** when its first event starts before the split date, and in the test set otherwise.
4. **A train case which spans the split**, that is, whose last event starts at or after the split date, stays in the train set with ``spanning_cases="train"`` (default), or is dropped with ``spanning_cases="drop"``. Dropping avoids training on events from the test period, at the cost of the cases around the split date.

The assignment is returned in ``data["cases"]``, one row per case of the log:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``log_caseid``
     - The ``caseid`` of the event-log
   * - ``num_events``
     - Number of events of the case
   * - ``start``, ``end``
     - Start of the first and of the last event of the case
   * - ``set``
     - ``train``, ``test`` or ``dropped``
   * - ``dropped_reason``
     - ``single event``, ``spans the split date``, or empty when the case is kept
   * - ``split_date``
     - The split date, the same in every row

The same table can be computed without building anything else, with ``split_cases``, which takes the same parameters. It is the way to evaluate several models on the same cases: compute the split once, give it to ``prefix_data`` as ``case_sets``, and use it to select the test cases in any other pipeline.

.. code-block:: python

    from SynBPS.dataprep.prepare import split_cases, prefix_data

    cases = split_cases(log, trainsize=0.7, split_by="cases", spanning_cases="drop")

    data = prefix_data(log, case_sets=cases)               # the same cases on each side of the split
    test_log = log[log.caseid.isin(cases.loc[cases.set == "test", "log_caseid"])]   # for another model

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

With ``include_end=True``, the last event of every trace is an observation as well: its next activity is END, which is the first column of ``y_a`` (``y_a_t1_1``), and its remaining time and time to the next event are 0. This is the setting for a model which also predicts whether the trace ends. A case with a single event then has one observation, when ``drop_single_event_cases`` is False.

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
     - The ``caseid`` of the event-log, to join with the log and with ``cases``
   * - ``num_events``
     - Number of events of the trace
   * - ``prefix_number``
     - Number of observed events in the observation, which is also the ``event_number`` of its last event
   * - ``prefixes``
     - Number of observations of the trace, ``num_events - 1`` (``num_events`` with ``include_end``)
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

All parameters
---------------
.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Meaning
   * - ``trainsize``
     - 0.7
     - Share of the cases (``split_by="cases"``) or of the days of the log (``split_by="date"``) in the train set
   * - ``split_by``
     - ``"cases"``
     - How the split date is computed from ``trainsize``, see `Train and test split`_
   * - ``spanning_cases``
     - ``"train"``
     - ``"train"`` or ``"drop"``: what happens to a train case whose last event starts at or after the split date
   * - ``split_date``
     - None
     - A timestamp or date string which replaces the computed split date
   * - ``case_sets``
     - None
     - A table from ``split_cases``, which replaces the split computed by ``prefix_data``
   * - ``drop_single_event_cases``
     - True
     - Drop the cases with a single event
   * - ``include_end``
     - False
     - Keep the last event of every trace as an observation, with END as its next activity
   * - ``category_cols``
     - []
     - Columns of the log added as one-hot features
   * - ``numeric_cols``
     - []
     - Columns of the log added as numeric features
   * - ``dummify_time_features``
     - False
     - One-hot encode ``dayofweek`` and ``hourofday``
   * - ``window_position``
     - ``"last_k"``
     - Which events are kept when a trace is longer than the prefix length (never the case in ``prefix_data``)
   * - ``scaling``
     - None
     - ``"standardize"`` or ``"normalize"`` the numeric features with the parameters of the train set
   * - ``verbose``
     - False
     - Print the progress

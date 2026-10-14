.. eventlog:


Generating an event-log
=========================
A single event-log is generated with ``generate_eventlog``, which takes a dictionary of settings and returns a pandas dataframe with one row per event. The settings are described on the :doc:`parameters` page. The same code is in the `event-log example notebook <https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example.ipynb>`_.

.. code-block:: python

    eventlog_settings = {
                    # number of traces/cases in the event-log
                    "number_of_traces":1000,

                    # level of entropy: min, medium and/or max
                    "process_entropy": "med_entropy",# "min_entropy","med_entropy","max_entropy"

                    # first or higher-order markov chain to represent the transitions "memoryless", "memory"
                    "process_type":"memory",#"memoryless",

                    # order of the HOMC - only specify this when using process with memory
                    "process_memory":2,

                    # minimum probability of ending the trace from any state - only used for process with memory
                    "p_abs_min":0.05,

                    # number of activity types
                    "statespace_size":5,
                    
                    # number of possible transitions from each state - only used for medium entropy (between 2 and statespace size, plus 1 for the process with memory)
                    "med_ent_n_transitions":3,
                                    
                    # mean time between arrivals of two traces, in days (alg. 1)
                    "inter_arrival_time":1.5,
                    
                    # mean of the exponential noise added before each event, in days (eq. 19). 0 means no noise
                    "process_stability_scale":0.1,
                    
                    # probability of agent being available
                    "resource_availability_p":0.5,

                    # number of agents in the process
                    "resource_availability_n":3,

                    # time per request until an agent is available, in days. 0.041 days is about 1 hour
                    "resource_availability_m":0.041,
                    
                    # upper limit of the mean activity durations Lambda(d, t), in days (eq. 18)
                    "activity_duration_lambda_range":1,
                    
                    # business hours definition: when can cases be processed? weekdays, all-week or none
                    "Deterministic_offset_W":"weekdays",

                    # time-unit for a full week: days = 7, hrs = 24*7, etc.
                    "Deterministic_offset_u":7,

                    # offset for the timestamps used (years after 1970)
                    "datetime_offset":54,

                    # seed value for reproducibility:
                    "seed_value":1337
                    }

    from SynBPS.simulation.simulate_eventlog import generate_eventlog

    log = generate_eventlog(eventlog_settings, verbose=True)

All settings are checked before the simulation starts. A missing or invalid setting raises a ``ValueError`` which lists every problem found, together with the setting to change.

The event-log
------------------
The returned dataframe has one row per event, sorted by trace and by position in the trace. Continuous time is measured in days from the start of the simulation, where t = 0 is Monday 00:00 of the first week. The timestamps are derived from it, such that the open hours of a day are 06:00 to 18:00: t = 0 is the first Sunday of the year ``1970 + datetime_offset`` at 18:00, and the timestamps are rounded to whole seconds.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning
   * - ``caseid``
     - Id of the trace (case), as a string
   * - ``activity``
     - Name of the activity, a letter
   * - ``activity_no``
     - Position of the event in the trace, starting at 1
   * - ``z_t``
     - Arrival time of the trace (alg. 1)
   * - ``n_t``
     - Time at which the event is ready to start: the arrival time plus the total durations of the previous events
   * - ``h_t``
     - Resource offset: waiting time until an agent is available (alg. 8)
   * - ``b_t``
     - Stability offset: noise added before the event starts (eq. 19)
   * - ``q_t``
     - Position in the week of the time at which the event would start without business hours
   * - ``s_t``
     - Deterministic offset: waiting time until the process is open again (business hours, alg. 9)
   * - ``v_t``
     - Duration of the activity itself, exponential with mean ``Lambda(activity, activity_no)``
   * - ``u_t``
     - Total duration of the event: ``h_t + b_t + s_t + v_t``
   * - ``y_acc_sum``
     - Accumulated total durations of the trace up to and including this event
   * - ``starttime``, ``endtime``
     - Start and end of the activity in continuous time
   * - ``arrival_datetime``, ``start_datetime``, ``end_datetime``
     - ``z_t``, ``starttime`` and ``endtime`` as timestamps, counted from the first Sunday at 18:00 of the year 1970 plus ``datetime_offset``
   * - ``start_day``, ``start_hour``
     - Weekday and hour of ``start_datetime``

The process with memory
-------------------------
With ``process_type = "memory"``, the next activity depends on the last ``process_memory`` activities of the trace (a higher-order Markov chain, alg. 7). The transition tables are generated from the seed and can be recovered without generating a log:

.. code-block:: python

    from SynBPS.simulation.simulation_helpers import make_D
    from SynBPS.simulation.alg7_memory_process_generator import create_homc

    # the same activity names as in the event-log
    D = make_D(eventlog_settings["statespace_size"])

    HOMC = create_homc(D,
                       K=eventlog_settings["process_memory"],
                       mode=eventlog_settings["process_entropy"],
                       n_transitions=eventlog_settings["med_ent_n_transitions"],
                       p_abs_min=eventlog_settings["p_abs_min"],
                       seed_value=eventlog_settings["seed_value"])

    # initial probabilities over the activities
    HOMC["P0"]

    # probability vector over the activities and the absorption state END, given the last two activities
    HOMC["Phi"][2][("a", "b")]

``HOMC["Phi"]`` holds one table per order 1 to k. ``Phi[i]`` maps every context of ``i`` activities to a probability vector over the activities plus the absorption state ``END``. The first events of a trace, which have fewer than k previous activities, use the table of their own length. In every context the probability of ``END`` is at least ``p_abs_min``, which guarantees that every trace ends.

With ``min_entropy`` the process is one deterministic trace, with ``med_entropy`` each context has ``med_ent_n_transitions`` possible next activities with random weights, and with ``max_entropy`` every transition is equally likely from every context. The levels form a ladder: ``min_entropy``, then ``med_entropy`` with 2 up to ``statespace_size`` + 1 transitions, then ``max_entropy``. ``max_entropy`` is the uniform extreme, where the next activity does not depend on the previous ones: ``process_memory`` has no effect on the control-flow at this level, and ``check_settings`` warns when it is above 1. The highest level where every transition is possible and the memory order matters is ``med_entropy`` with ``med_ent_n_transitions`` = ``statespace_size`` + 1.

Reproducibility
------------------
Every random draw of the simulation is derived from ``seed_value``. Each component (the transition tables, the sampling of the traces, the arrival times, the duration parameters, the resource offsets, the stability offsets and the activity durations) has its own random stream, for both process types, and the global stream of numpy is not used. This gives the following properties:

* Two calls with the same settings give the same event-log, in every column.
* The first traces of an event-log do not change when ``number_of_traces`` is increased: a larger log is a larger sample from the same process.
* Changing one setting only changes the draws of the component which uses it. For example, changing ``resource_availability_p`` leaves the activities, the arrival times, the stability offsets and the activity durations unchanged, and only changes the resource offsets ``h_t`` and everything derived from them.
* Random draws made by other code, such as ``numpy.random`` calls in a model or in a library, do not change the event-log, and a simulation does not change the state of ``numpy.random``.
* The process behind a log can be recovered from its settings without generating the log, see `The data-generating process`_ below.

The data-generating process
----------------------------
The process behind an event-log is known, which is the point of a simulation in a controlled study: with the initial probabilities, the transition tables and the mean activity durations in hand, the next activity with the highest probability can be read off the tables, and the remaining time can be simulated from the true parameters. ``recover_dgp`` returns them from the settings, without generating the log, and ``generate_eventlog`` returns them together with the log with ``return_dgp=True``. Both give the same process.

.. code-block:: python

    from SynBPS.simulation.simulate_eventlog import generate_eventlog, recover_dgp

    dgp = recover_dgp(eventlog_settings)                          # from the settings
    log, dgp = generate_eventlog(eventlog_settings, return_dgp=True)   # together with the log

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``process_type``, ``mode``
     - ``process_type`` and ``process_entropy`` of the settings
   * - ``D``, ``D_abs``
     - The activities, and the activities followed by the absorption state ``END``
   * - ``K``
     - The order of the process: ``process_memory``, or 1 for the memoryless process
   * - ``P0``
     - The initial probabilities, an array over ``D``
   * - ``Phi``
     - The transition tables: ``Phi[k][context]`` is the probability vector over ``D_abs`` given the last ``k`` activities (a tuple), for ``k`` from 1 to ``K``. A prefix shorter than ``K`` uses the table of its own length. The memoryless process has the one table ``Phi[1]``, with the rows of its transition matrix
   * - ``P``
     - The transition matrix of the memoryless process as a dataframe with the rows and columns ``D_abs`` (``None`` for the process with memory)
   * - ``Lambda``
     - The mean activity durations in days: one row per position in the trace (0 for the first event) and one column per activity, for every position up to ``max_len``
   * - ``time_settings``
     - The settings of the time variables: ``inter_arrival_time``, ``process_stability_scale``, ``resource_availability_p``, ``_n``, ``_m`` and ``_mode``, ``activity_duration_lambda_range``, ``Deterministic_offset_W`` (as a matrix of closed hours) and ``Deterministic_offset_u``
   * - ``max_len``, ``seed_value``, ``datetime_offset``
     - The settings of the same name

The next activity with the highest probability after the observed activities of a prefix, for either process type:

.. code-block:: python

    def most_likely_next_activity(dgp, activities):
        k = min(len(activities), dgp["K"])
        p = dgp["Phi"][k][tuple(activities[-k:])]
        return dgp["D_abs"][p.argmax()]        # "END" when the trace most likely ends

The time variables of an event are the resource offset (geometric number of requests times ``resource_availability_m``), the stability offset (exponential with mean ``process_stability_scale``), the business-hours offset (deterministic, from the time in the week) and the duration (exponential with mean ``Lambda`` at the activity and position). The expected remaining time of a prefix therefore depends on the time of its last event, and is computed by simulating the continuation of the trace from ``Phi`` and ``Lambda`` with these offsets.

Custom distributions
----------------------
For the memoryless process, the distributions can be given as csv files instead of being generated. Set ``process_entropy`` to ``"custom"``, ``process_type`` to ``"memoryless"``, and add the files to the settings. See the `custom distribution example notebook <https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example_custom_dist.ipynb>`_ and the files in ``examples/data``.

.. code-block:: python

    eventlog_settings["process_type"] = "memoryless"
    eventlog_settings["process_entropy"] = "custom"
    eventlog_settings["custom_distributions"] = {"p0":"data/p0.txt",
                                                 "p":"data/p.txt",
                                                 "Lambda":"data/lambda.txt"}

    log = generate_eventlog(eventlog_settings)

The files have the following layout. The column names of the ``p`` and ``Lambda`` files are ignored and replaced by the activity names, so only the shapes matter.

* **p0**: the initial probabilities, one row per activity in a column named ``p0``. ``statespace_size`` rows.
* **p**: the transition matrix. Rows are the activity the transition comes from, columns the activity it goes to, and the last row and column are the absorption state. ``statespace_size + 1`` rows and columns; every row sums to 1, and the last row has 1 in the last column.
* **Lambda**: the mean activity durations, one row per position in the trace and one column per activity. At least as many rows as the longest trace plus one (for the absorption state); an error names the file if it is too short.

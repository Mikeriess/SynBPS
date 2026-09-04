.. eventlog:


Generating an event-log
=========================
A single event-log is generated with ``generate_eventlog``, which takes a dictionary of settings and returns a pandas dataframe with one row per event. The settings are described on the :doc:`parameters` page. The same code is in the `event-log example notebook <https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example.ipynb>`_.

.. code-block:: python

    eventlog_settings = {
                    # number of traces/cases in the event-log
                    "number_of_traces":1000,

                    # level of entropy: min, medium and/or max
                    "process_entropy": "max_entropy",# "min_entropy","med_entropy","max_entropy"

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
                                    
                    # lambda parameter of inter-arrival times
                    "inter_arrival_time":1.5,
                    
                    # lambda parameter of process noise
                    "process_stability_scale":0.1,
                    
                    # probability of agent being available
                    "resource_availability_p":0.5,

                    # number of agents in the process
                    "resource_availability_n":3,

                    # waiting time in full days per request, when no agent is available. 0.041 days is about 1 hour
                    "resource_availability_m":0.041,
                    
                    # variation between activity durations
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
The returned dataframe has one row per event, sorted by trace and by position in the trace. Continuous time is measured in days from the start of the simulation; the timestamps are derived from it, such that the open hours of a day are 06:00 to 18:00.

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
     - ``z_t``, ``starttime`` and ``endtime`` as timestamps, in the year 1970 plus ``datetime_offset``
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
    HOMC["Phi"][2][("b", "c")]

``HOMC["Phi"]`` holds one table per order 1 to k. ``Phi[i]`` maps every context of ``i`` activities to a probability vector over the activities plus the absorption state ``END``. The first events of a trace, which have fewer than k previous activities, use the table of their own length. In every context the probability of ``END`` is at least ``p_abs_min``, which guarantees that every trace ends.

With ``min_entropy`` the process is one deterministic trace, with ``med_entropy`` each context has ``med_ent_n_transitions`` possible next activities, and with ``max_entropy`` every transition is equally likely from every context (so ``process_memory`` has no effect on the control-flow at this level).

Reproducibility
------------------
Every random draw of the simulation is derived from ``seed_value``. Each component (the transition tables, the sampling of the traces, the arrival times, the duration parameters, the resource offsets, the stability offsets and the activity durations) has its own random stream, which gives the following properties:

* Two calls with the same settings give the same event-log, in every column.
* The first traces of an event-log do not change when ``number_of_traces`` is increased: a larger log is a larger sample from the same process.
* Changing one setting only changes the draws of the component which uses it. For example, changing ``resource_availability_p`` leaves the activities, the arrival times, the stability offsets and the activity durations unchanged, and only changes the resource offsets ``h_t`` and everything derived from them.

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

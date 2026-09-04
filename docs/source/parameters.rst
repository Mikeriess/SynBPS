Settings
===================
All settings of ``generate_eventlog`` and of a design table, with the name used in the paper, the unit and an example value. Time is measured in days. The last column says which process type or entropy level uses the setting.

.. list-table::
   :header-rows: 1
   :widths: 18 14 40 12 10 10

   * - Setting
     - In the paper
     - Meaning
     - Unit
     - Example
     - Used by
   * - ``number_of_traces``
     - n
     - Number of traces (cases) in the event-log
     - count
     - 1000
     - both
   * - ``process_type``
     - process without / with memory
     - ``memoryless``: first-order Markov chain (alg. 6). ``memory``: higher-order Markov chain (alg. 7)
     - -
     - ``"memory"``
     - both
   * - ``process_entropy``
     - entropy level
     - ``min_entropy``: one deterministic trace. ``med_entropy``: ``med_ent_n_transitions`` possible transitions from each state. ``max_entropy``: every transition equally likely (alg. 4). ``custom``: distributions from files, see ``custom_distributions``
     - -
     - ``"med_entropy"``
     - both
   * - ``process_memory``
     - k
     - Order of the higher-order Markov chain: the number of previous activities the next activity depends on
     - count, 1 or more
     - 2
     - memory
   * - ``statespace_size``
     - d_size
     - Number of activity types. The activities are named with letters
     - count, 2 to 27
     - 5
     - both
   * - ``med_ent_n_transitions``
     - n_transitions
     - Number of possible transitions from each state (context). Between 2 and ``statespace_size``, plus 1 for the process with memory, where the absorption state can be one of them
     - count
     - 3
     - med_entropy
   * - ``p_abs_min``
     - not in the paper (added in 1.1.8)
     - Minimum probability of ending the trace from any context, which guarantees that every trace ends. 0 removes the guarantee
     - probability, 0 to 1 (1 excluded)
     - 0.05
     - memory, med_entropy and max_entropy
   * - ``max_len``
     - not in the paper (added in 1.1.8)
     - Maximum number of events in a trace, after which an exception is raised
     - count
     - 10000
     - memory
   * - ``inter_arrival_time``
     - inter-arrival time (alg. 1)
     - Mean time between the arrivals of two traces: the scale of the exponential distribution, not a rate
     - days
     - 1.5
     - both
   * - ``process_stability_scale``
     - process stability (eq. 19)
     - Mean of the exponential noise added before each event. 0 means no noise
     - days
     - 0.1
     - both
   * - ``resource_availability_p``
     - p (alg. 8)
     - Probability that an agent is available at a request
     - probability, above 0 and at most 1
     - 0.5
     - both
   * - ``resource_availability_n``
     - n (alg. 8)
     - Number of agents
     - count, 1 or more
     - 3
     - both
   * - ``resource_availability_m``
     - m (alg. 8)
     - Waiting time per request until an agent is available. The number of requests is geometric with success probability 1 - (1 - p)^n. Note that 0.041 days is about one hour, and 15 minutes is 0.0104 days
     - days
     - 0.041
     - both
   * - ``activity_duration_lambda_range``
     - xi
     - Upper limit of the uniform distribution (from 0.0001) the mean activity durations Lambda(d, t) are drawn from, one per activity and position in the trace. The duration of an event is exponential with that mean (the scale parameter of numpy, not a rate)
     - days
     - 1
     - both
   * - ``Deterministic_offset_W``
     - deterministic offset (alg. 9)
     - Business hours. ``weekdays``: open 06:00 to 18:00, Monday to Friday. ``all-week``: open 06:00 to 18:00 every day. ``none``: always open. An event which is due during closed hours starts when the process opens again; its duration then runs uninterrupted
     - -
     - ``"weekdays"``
     - both
   * - ``Deterministic_offset_u``
     - eta (eq. 21)
     - Length of a week in the time unit used: 7 when the time unit is days. The business hours above assume 7
     - time units per week
     - 7
     - both
   * - ``datetime_offset``
     - not in the paper
     - Years added to 1970 for the timestamps
     - years
     - 54
     - both
   * - ``seed_value``
     - seed
     - Seed of all random streams. The same settings and seed give the same event-log, and the first traces of a log do not change when ``number_of_traces`` grows
     - integer, 0 to 2^32 - 1
     - 1337
     - both
   * - ``custom_distributions``
     - not in the paper
     - Files with the initial probabilities ``p0``, the transition matrix ``p`` and the duration parameters ``Lambda``, for ``process_entropy`` ``custom``. See the custom distribution example notebook
     - -
     - ``{"p0": "data/p0.csv", "p": "data/p.csv", "Lambda": "data/lambda.csv"}``
     - memoryless, custom

Properties of the simulation
-----------------------------

* Traces are independent of each other: the resource offset is drawn per event (alg. 8), and there is no shared queue.
* The mean duration ``Lambda(d, t)`` depends on the activity and on its position in the trace, so the same activity has a different mean duration at position 2 and position 5.
* The transition tables of the process with memory can be recovered from the seed with ``create_homc`` (``SynBPS.simulation.alg7_memory_process_generator``), independent of ``number_of_traces``. A prefix shorter than ``process_memory`` uses the table of its own length.
* ``number_of_traces`` is not the number of events: the trace length depends on the entropy level, ``statespace_size``, ``med_ent_n_transitions``, ``p_abs_min`` and ``process_memory``.
* ``run_experiments`` does not pass the settings to the training function. A training function which needs the data generating process must recover it from the seed.

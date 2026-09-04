# SynBPS
[![Downloads](https://static.pepy.tech/badge/synbps)](https://pepy.tech/project/synbps) [![Documentation Status](https://readthedocs.org/projects/synbps/badge/?version=latest)](https://synbps.readthedocs.io/en/latest/?badge=latest)


SynBPS is short for Synthetic Business Process Simulation. This framework is designed to simulate **synthetic** business processes. In a nutshell, this framework lets you run predictive process monitoring experiments across **multiple business processes**, specified by well-known parametric distributions. See more in the publication: [Riess (2024)](https://journals.sagepub.com/doi/abs/10.1177/00375497241233326) [[pdf](https://journals.sagepub.com/doi/pdf/10.1177/00375497241233326?casa_token=h9BOK2WWdQQAAAAA:t46xt6_qhz651cLzDVktuPnr3ku-eRaWNk9vECyHEAZsl3OtUHCffCZncn48XI0BprdrZM8VcBT3)]


![image](https://github.com/Mikeriess/SynBPS/blob/main/docs/illustration.png)

## Whats new: Version 1.1.9
- Fixed the resource offset ```h_t``` (and with it all timestamps) not being reproducible from ```seed_value```. The waiting time is now drawn from the geometric distribution of the number of requests until an agent is available, which has the same distribution as the previous loop of binomial draws
- Fixed ```med_ent_n_transitions``` being ignored by the memoryless process (5 transitions were always used, and a ```statespace_size``` below 5 stopped the interpreter)
- Fixed the duration parameters ```Lambda``` changing with the longest trace, such that a log with more traces from the same seed had other duration parameters
- Fixed invalid settings raising a ```TypeError``` or stopping the interpreter. All settings are now checked at the start of ```generate_eventlog```, and a ```ValueError``` lists every invalid setting
- Fixed unknown values of ```Deterministic_offset_W``` raising an ```UnboundLocalError```, and the first closed interval of the business hours starting at 0.001 instead of 0
- Added ```Deterministic_offset_W = "none"``` for a process without closed hours
- Added one independent random stream per component: arrival times, duration parameters, resource offsets, stability offsets, activity durations, and the tables and the sampling of the process with memory. Changing one setting no longer changes the draws of the other components, and the first traces of a log are identical when ```number_of_traces``` is increased
- Changed ```max_entropy``` for the process with memory to equal transition probabilities from every context, as in the memoryless process (alg. 4). With ```max_entropy```, ```process_memory``` therefore no longer changes the control-flow
- Added tests for reproducibility, the settings check and the business hours

**Please note:** Durations and timestamps, and the control-flow of the process with memory, differ numerically from version 1.1.8 and earlier for the same seed value. The control-flow of the memoryless process is unchanged for ```min_entropy```, ```max_entropy``` and ```custom```, and differs for ```med_entropy``` because ```med_ent_n_transitions``` is now used.

## Whats new: Version 1.1.8
- Fixed the process with memory (HOMC): there is now one transition table per order 1 to K, and every table is conditioned on the full context of previous activities. Before, the effective order was 1 for ```process_memory``` 2, and other orders raised an error
- Fixed the conditional probabilities of the process with memory, such that they sum to 1 within every context
- Fixed ```med_ent_n_transitions``` being ignored by the process with memory
- Fixed the initial probabilities of the process with memory, such that a trace cannot start in the absorption state
- Fixed an endless loop in the process with memory, when no state led to the absorption state
- Added ```p_abs_min```: the minimum probability of ending the trace from any state of the process with memory (default 0.05), which guarantees that every trace ends
- Added ```min_entropy``` for the process with memory, as a deterministic process of order K. Before, this setting silently used the memoryless process
- Added the transition tables of the process with memory as a plain dictionary ```Phi[order][context]```, which can be inspected and stored directly
- Added tests for the process with memory
- Removed the ```Memory_process``` module (ported from Pomegranate) and the ```networkx``` dependency. Generating a process with memory is now much faster, as the sampling no longer loops over the full table for every event

**Please note:** Event-logs generated with ```process_type = "memory"``` in version 1.1.8 differ from earlier versions for the same seed value, as the earlier versions did not produce a process of order K.

## Whats new: Version 1.1.3
- Added support for process memory with HOMC of order > 4
- Added Example notebooks in ```examples/``` folder
- Added ability to specify distribution parameters (memoryless process)
- Added ability to specify the dataprep function manually (see [e2e example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/simulation_e2e_example.ipynb))
- Fixed issues with seed value in processes with memory
- Restructuring and separation of functions, based on their purpose: 
	- ```Design``` for generating a DoE
	- ```Simulation``` for functions related to event-log generation
	- ```Dataprep``` for functions related to data-preparation for ML models (prefix-log, temporal splitting etc.,)
- Updated readthedocs documentation with version *1.1.0+* syntax changes.
- Other minor fixes

**Please note:** Version 1.1.0** introduces new parameters and different function locations. Users are therefore advised to refer to the slightly changed code examples in ```examples/``` folder.

# Getting Started
You can install SynBPS using pip:

    pip install SynBPS

### Installing a specific version from GitHub
A release can also be installed directly from GitHub, for example when a new release is not yet available on PyPI:

    pip install git+https://github.com/Mikeriess/SynBPS.git@v1.1.8

Replace ```v1.1.8``` with the tag of the release you need (see the [releases page](https://github.com/Mikeriess/SynBPS/releases)). This requires git to be installed on your machine. Each release also carries the built wheel file (```.whl```) as an asset, which can be installed with ```pip install <path or URL of the wheel file>``` without git.

Once installed, you can:

- Run a simulation experiment with your own models using the [End-to-end example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/simulation_e2e_example.ipynb) for a short demo of SynBPS. 
- Or simply generate a single event-log using the example code in the [Event-log example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example.ipynb). This code example also lets you integrate the power of SynBPS into your own custom code pipeline (for advanced users).
- For the memoryless process, you can also specify the parameters of the distributions manually as shown in the [Custom distribution Event-log example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example_custom_dist.ipynb).


## Settings
All settings of ```generate_eventlog``` and of a design table, with the name used in the paper, the unit and an example value. Time is measured in days. The last column says which process type or entropy level uses the setting.

| Setting | In the paper | Meaning | Unit | Example | Used by |
|---|---|---|---|---|---|
| `number_of_traces` | n | Number of traces (cases) in the event-log | count | 1000 | both |
| `process_type` | process without / with memory | `memoryless`: first-order Markov chain (alg. 6). `memory`: higher-order Markov chain (alg. 7) |  | `"memory"` | both |
| `process_entropy` | entropy level | `min_entropy`: one deterministic trace. `med_entropy`: `med_ent_n_transitions` possible transitions from each state. `max_entropy`: every transition equally likely (alg. 4). `custom`: distributions from files, see `custom_distributions` |  | `"med_entropy"` | both |
| `process_memory` | k | Order of the higher-order Markov chain: the number of previous activities the next activity depends on | count, 1 or more | 2 | memory |
| `statespace_size` | d_size | Number of activity types. The activities are named with letters | count, 2 to 27 | 5 | both |
| `med_ent_n_transitions` | n_transitions | Number of possible transitions from each state (context). Between 2 and `statespace_size`, plus 1 for the process with memory, where the absorption state can be one of them | count | 3 | med_entropy |
| `p_abs_min` | not in the paper (added in 1.1.8) | Minimum probability of ending the trace from any context, which guarantees that every trace ends. 0 removes the guarantee | probability, 0 to 1 (1 excluded) | 0.05 | memory, med_entropy and max_entropy |
| `max_len` | not in the paper (added in 1.1.8) | Maximum number of events in a trace, after which an exception is raised | count | 10000 | memory |
| `inter_arrival_time` | inter-arrival time (alg. 1) | Mean time between the arrivals of two traces: the scale of the exponential distribution, not a rate | days | 1.5 | both |
| `process_stability_scale` | process stability (eq. 19) | Mean of the exponential noise added before each event. 0 means no noise | days | 0.1 | both |
| `resource_availability_p` | p (alg. 8) | Probability that an agent is available at a request | probability, above 0 and at most 1 | 0.5 | both |
| `resource_availability_n` | n (alg. 8) | Number of agents | count, 1 or more | 3 | both |
| `resource_availability_m` | m (alg. 8) | Waiting time per request until an agent is available. The number of requests is geometric with success probability 1 - (1 - p)^n. Note that 0.041 days is about one hour, and 15 minutes is 0.0104 days | days | 0.041 | both |
| `activity_duration_lambda_range` | xi | Upper limit of the uniform distribution (from 0.0001) the mean activity durations Lambda(d, t) are drawn from, one per activity and position in the trace. The duration of an event is exponential with that mean (the scale parameter of numpy, not a rate) | days | 1 | both |
| `Deterministic_offset_W` | deterministic offset (alg. 9) | Business hours. `weekdays`: open 06:00 to 18:00, Monday to Friday. `all-week`: open 06:00 to 18:00 every day. `none`: always open. An event which is due during closed hours starts when the process opens again; its duration then runs uninterrupted |  | `"weekdays"` | both |
| `Deterministic_offset_u` | eta (eq. 21) | Length of a week in the time unit used: 7 when the time unit is days. The business hours above assume 7 | time units per week | 7 | both |
| `datetime_offset` | not in the paper | Years added to 1970 for the timestamps | years | 54 | both |
| `seed_value` | seed | Seed of all random streams. The same settings and seed give the same event-log, and the first traces of a log do not change when `number_of_traces` grows | integer, 0 to 2^32 - 1 | 1337 | both |
| `custom_distributions` | not in the paper | Files with the initial probabilities `p0`, the transition matrix `p` and the duration parameters `Lambda`, for `process_entropy` `custom`. See the custom distribution example notebook |  | `{"p0": "data/p0.csv", "p": "data/p.csv", "Lambda": "data/lambda.csv"}` | memoryless, custom |

Some properties of the simulation which are useful to know when building on SynBPS:

- Traces are independent of each other: the resource offset is drawn per event (alg. 8), and there is no shared queue.
- The mean duration ```Lambda(d, t)``` depends on the activity and on its position in the trace, so the same activity has a different mean duration at position 2 and position 5.
- The transition tables of the process with memory can be recovered from the seed with ```create_homc``` (```SynBPS.simulation.alg7_memory_process_generator```), independent of ```number_of_traces```. A prefix shorter than ```process_memory``` uses the table of its own length.
- ```number_of_traces``` is not the number of events: the trace length depends on the entropy level, ```statespace_size```, ```med_ent_n_transitions```, ```p_abs_min``` and ```process_memory```.
- ```run_experiments``` does not pass the settings to the training function. A training function which needs the data generating process must recover it from the seed.

## Documentation
See the [official documentation here](https://synbps.readthedocs.io/en/latest/).


## Citation
If you use SynBPS, please cite the corresponding paper. The paper can be cited as:

```
@article{riess2024synbps,
	title={SynBPS: a parametric simulation framework for the generation of event-log data},
	author={Riess, Mike},
	journal={SIMULATION},
	pages={00375497241233326},
	year={2024},
	publisher={SAGE Publications Sage UK: London, England}
}
```

## Contributing
If you would like to contribute to SynBPS, you are welcome to submit your suggestions, bug reports, or pull requests. Follow [the guidelines](https://github.com/Mikeriess/SynBPS/blob/main/src/contributing.md) to ensure smooth collaboration.


## Thanks
Jacob Schreiber and Pomegranate team. Joachim Scholderer and Kristoffer Lien.
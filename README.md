# SynBPS
[![Downloads](https://static.pepy.tech/badge/synbps)](https://pepy.tech/project/synbps) [![Documentation Status](https://readthedocs.org/projects/synbps/badge/?version=latest)](https://synbps.readthedocs.io/en/latest/?badge=latest)


SynBPS is short for Synthetic Business Process Simulation. This framework is designed to simulate **synthetic** business processes. In a nutshell, this framework lets you run predictive process monitoring experiments across **multiple business processes**, specified by well-known parametric distributions. See more in the publication: [Riess (2024)](https://journals.sagepub.com/doi/abs/10.1177/00375497241233326) [[pdf](https://journals.sagepub.com/doi/pdf/10.1177/00375497241233326?casa_token=h9BOK2WWdQQAAAAA:t46xt6_qhz651cLzDVktuPnr3ku-eRaWNk9vECyHEAZsl3OtUHCffCZncn48XI0BprdrZM8VcBT3)]

## Whats new: Version 1.1.0
- Added support for process memory with HOMC of order > 4
- Added Example notebooks in ```examples/``` folder
- Fixed issues with seed value in processes with memory
- Restructuring and separation of functions, based on their purpose: 
	- ```Design``` for generating a DoE
	- ```Simulation``` for functions related to event-log generation
	- ```Dataprep``` for functions related to data-preparation for ML models (prefix-log, temporal splitting etc.,)
- Removed dependency on Cython
- Other minor fixes

**Please note:** Version 1.1.0** introduces new parameters and different function locations. Users are therefore advised to refer to the example notebooks in ```examples/``` folder for examples.

# Getting Started
You can install SynBPS using pip:

    pip install SynBPS

Once installed, you can:

- Run a simulation experiment with your own models using the [End-to-end example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/simulation_e2e_example.ipynb) for a short demo of SynBPS. 
- Or simply generate a single event-log using the example code in the [Event-log example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example.ipynb). This code example also lets you integrate the power of SynBPS into your own custom code pipeline (for advanced users).


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

## How?
The intended usage of this framework is to benchmark new methods within predictive process monitoring. Rather than calibrating a simulation model from an existing process (as in existing frameworks), the aim is to simulate theoretical processes with varying degrees of noise in duration distributions and entropy in the control-flow. The framework uses algorithms (described in the publication), Higher-order Markov Chains (HOMC) and the Hypo-exponential distribution to represent temporal dependency (or its absence) in conditional duration distributions. 

![image](https://github.com/Mikeriess/SynBPS/blob/main/docs/illustration.png)

The benefit of SynBPS is in the transparency (and simplicity) of the data generating process, which can help further understand the influence of process characteristics on predictive performance. By e.g. changing the entropy of the process, SynBPS lets you compare the difference in predictive performance across everything between predictable to completely chaotic processes.


## Contributing
If you would like to contribute to SynBPS, you are welcome to submit your suggestions, bug reports, or pull requests. Follow [the guidelines](https://github.com/Mikeriess/SynBPS/blob/main/src/contributing.md) to ensure smooth collaboration.


## Thanks
Jacob Schreiber and Pomegranate team. Joachim Scholderer and Kristoffer Lien.
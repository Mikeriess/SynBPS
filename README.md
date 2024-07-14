# SynBPS
[![Downloads](https://static.pepy.tech/badge/synbps)](https://pepy.tech/project/synbps) [![Documentation Status](https://readthedocs.org/projects/synbps/badge/?version=latest)](https://synbps.readthedocs.io/en/latest/?badge=latest)

SynBPS is short for Synthetic Business Process Simulation. The framework is designed to simulate **synthetic** business processes. In a nutshell, this framework lets you run experiments across **multiple business processes**, specified by well-known parametric distributions. See more in the publication: [Riess (2024)](https://journals.sagepub.com/doi/abs/10.1177/00375497241233326)

## Whats new: Version 1.1.0
- Example notebooks in ```examples/``` folder
- Removed dependency on Cython
- Minor fixes

# Getting Started
You can install SynBPS using pip:

    pip install SynBPS

SynBPS requires pomegranate 0.14.8 and python 3.9 or higher.

## Example usage
See the [End-to-end example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/simulation_e2e_example.ipynb) for a short demo of SynBPS. Also, please refer to the [Event-log example notebook](https://github.com/Mikeriess/SynBPS/blob/main/examples/event_log_example.ipynb) for an example of how to simulate a single event-log, if you wish to implement this functionality in your own pipeline.

## Documentation
See the [official documentation here](https://synbps.readthedocs.io/en/latest/).

# Roadmap
- Extend HOMC to include h > 4 
- Add functionality to specify sampling approach of HOMC
- Add data pre-processing in more formats



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
The intended usage of this software is to benchmark new methods within predictive process monitoring research. Rather than calibrating a simulation model from an existing process (as in existing frameworks), the aim is to simulate theoretical processes with varying degrees of noise in duration distributions and entropy in the control-flow. The framework uses algorithms (described in the publication), Higher-order Markov Chains (HOMC) and the Hypo-exponential distribution to represent temporal dependency (or its absence) in conditional duration distributions. 

![image](https://github.com/Mikeriess/SynBPS/blob/main/docs/illustration.png)

## Why?
The benefit of SynBPS is in the transparency (and simplicity) of the data generating process, which can help further understand the influence of process characteristics on predictive performance. By e.g. changing the entropy of the process, SynBPS lets you compare the difference in predictive performance across everything between predictable to completely chaotic processes.


## Contributing
If you would like to contribute to SynBPS, you are welcome to submit your suggestions, bug reports, or pull requests. Follow the guidelines below to ensure smooth collaboration:

- Before submitting a new feature request or bug report, please check the existing issues to avoid duplicates.
- If you have a new feature idea, open an issue to discuss it with the maintainers and get feedback.
- For bug reports, provide a clear and concise description of the issue, including steps to reproduce it.
- If your contribution requires documentation changes, please update the documentation accordingly.
- Be respectful and considerate towards others in your interactions on the project.


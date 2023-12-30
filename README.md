# SynBPS
SynBPS is short for Synthetic Business Process Simulation, as it is intended for the simulation of **synthetic** (i.e. *multiple*, *hypothetical*) business processes from parametric distributions.

The intended usage of this software is to benchmark models within predictive process monitoring research. It is not intended for the simulation of real-world business processes, but rather as an addition to existing benchmark data, such as the BPI Challenge datasets. 

The benefit of SynBPS is the full transparency of the data generating process, which can help further understand the effects of different process characteristics on predictive performance. 

# Getting Started
You can install SynBPS using pip::

    pip install SynBPS

SynBPS requires python 3.9 or higher.

## Example usage
See the [example notebook](https://github.com/Mikeriess/SynBPS/blob/main/tests/test_pypi.ipynb) for a demonstration of the usage of SynBPS.

# Todos
- Extend HOMC to include h > 4
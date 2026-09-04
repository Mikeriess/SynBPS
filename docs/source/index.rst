.. SynBPS documentation master file, created by
   sphinx-quickstart on Sat Dec 30 15:28:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :caption: Getting Started
   :hidden:

   eventlog
   example
   parameters

Welcome to SynBPS's documentation!
===================================
SynBPS is short for Synthetic Business Process Simulation, as it is intended for the simulation of **synthetic** (i.e. *multiple*, *hypothetical*) business processes from a specified distribution of business processes.

The intended usage of this framework is to benchmark new methods within predictive process monitoring research. Rather than calibrating a simulation model from an existing process (as in existing frameworks), the aim is to simulate theoretical processes with varying degrees of noise in duration distributions and entropy in the control-flow. 

The benefit of SynBPS is in the full transparency of the data generating process, which can help further understand the influence of process characteristics on predictive performance. By changing the entropy of the process, SynBPS lets you compare the difference in predictive performance across everything between predictable to completely chaotic processes.

A process is simulated in two parts, as described in the paper (see the citation below):

* The **control-flow** (which activities occur, and in which order) is a first-order Markov chain (``process_type = "memoryless"``) or a higher-order Markov chain of order k (``process_type = "memory"``), whose transition probabilities are generated for a chosen level of entropy.
* The **timing** (arrival of each trace, waiting for a resource, noise, business hours and the duration of each activity) is generated from parametric distributions on top of the control-flow.

Every random draw is derived from a single ``seed_value``, so a design table with seeds fully determines the generated event-logs.


Getting Started
================
You can install SynBPS using pip::

    pip install SynBPS

A specific release can also be installed directly from GitHub, for example when a new release is not yet available on PyPI (this requires git)::

    pip install git+https://github.com/Mikeriess/SynBPS.git@v1.1.9

Then continue with one of the following pages:

* :doc:`eventlog`: generate a single event-log from a dictionary of settings, and inspect it.
* :doc:`example`: run a set of experiments from a design table, with your own data preparation, model and evaluation.
* :doc:`parameters`: all settings, with the names and units used in the paper.

The example notebooks in the `examples folder on GitHub <https://github.com/Mikeriess/SynBPS/tree/main/examples>`_ contain the same code as these pages.

The changes in each version are listed in the `README on GitHub <https://github.com/Mikeriess/SynBPS#whats-new-version-119>`_ and on the `releases page <https://github.com/Mikeriess/SynBPS/releases>`_. Note that event-logs generated with version 1.1.9 differ from those of earlier versions for the same seed value, as several parts of the simulation were not reproducible from the seed before.


Citation
-----------------

If you use SynBPS, please cite the corresponding paper. The paper can be cited as:
::

		@article{riess2024synbps,
		title={SynBPS: a parametric simulation framework for the generation of event-log data},
		author={Riess, Mike},
		journal={SIMULATION},
		pages={00375497241233326},
		year={2024},
		publisher={SAGE Publications Sage UK: London, England}
		}

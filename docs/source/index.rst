.. SynBPS documentation master file, created by
   sphinx-quickstart on Sat Dec 30 15:28:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SynBPS's documentation!
===================================
SynBPS is short for Synthetic Business Process Simulation, as it is intended for the simulation of **synthetic** (i.e. *multiple*, *hypothetical*) business processes from parametric distributions.

The intended usage of this software is to benchmark models within predictive process monitoring research. It is not intended for the simulation of real-world business processes, but rather as an addition to existing benchmark data, such as the BPI Challenge datasets. 

The benefit of SynBPS is the full transparency of the data generating process, which can help further understand the effects of different process characteristics on predictive performance. 

Getting Started
================

.. toctree::
   :caption: Getting Started
   :hidden:

   example
   installation

:doc:`example`
   SynBPS is designed to be used in the following manner:

   1. Generate design table (table of all settings to be simulated)
   2. Specify Train() and Test() functions
   3. Run experiments
   4. Analyze results

:doc:`installation`
   you can install SynBPS using pip with::

      pip install SynBPS



Citation
-----------------

If you use SynBPS, please cite the corresponding paper.

The paper can be cited as:
::

	@article{riess2024,
		  title={A parametric simulation framework for the generation of event-log data},
		  author={Riess, Mike},
		  journal={Pending},
		  volume={999},
		  number={999},
		  pages={999--999},
		  year={2024}
		}

Alternatively, the GitHub repository can be cited as:
::

	@misc{riess2023,
		author = {Mike Riess},
		title = {SynBPS},
		year = {2023},
		publisher = {GitHub},
		journal = {GitHub repository},
		howpublished = {\url{https://github.com/mikeriess/synbps}},
		commit = {enter commit that you used}
	}
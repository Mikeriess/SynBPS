.. SynBPS documentation master file, created by
   sphinx-quickstart on Sat Dec 30 15:28:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :caption: Getting Started
   :hidden:

   example

Welcome to SynBPS's documentation!
===================================
SynBPS is short for Synthetic Business Process Simulation, as it is intended for the simulation of **synthetic** (i.e. *multiple*, *hypothetical*) business processes from a specified distribution.

The intended usage of this software is to benchmark new methods within predictive process monitoring research. Rather than calibrating a simulation model from an existing process (as in existing frameworks), the aim is to simulate theoretical processes with varying degrees of noise in duration distributions and entropy in the control-flow. 

The benefit of SynBPS is in the full transparency of the data generating process, which can help further understand the influence of process characteristics on predictive performance. By changing the entropy of the process, SynBPS lets you compare the difference in predictive performance across everything between predictable to completely chaotic processes.


Getting Started
================
You can install SynBPS using pip::

    pip install SynBPS

SynBPS requires python 3.10 or higher.

Citation
-----------------

If you use SynBPS, please cite the corresponding paper. The paper can be cited as:
::

		@inbook{Riess2023Framework,
		author = {Riess, Mike},
		title = {A Parametric Simulation Framework for the Generation of Event-Log Data},
		booktitle = {Essays on Predictive and Prescriptive Process Monitoring},
		publisher = {Norwegian University of Life Sciences},
		year = {2023},
		pages = {75-98},
		}
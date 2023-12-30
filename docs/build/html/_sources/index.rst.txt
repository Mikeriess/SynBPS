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
SynBPS is short for Synthetic Business Process Simulation, as it is intended for the simulation of **synthetic** (i.e. *multiple*, *hypothetical*) business processes from parametric distributions.

The intended usage of this software is to benchmark models within predictive process monitoring research. It is not intended for the simulation of real-world business processes, but rather as an addition to existing benchmark data, such as the BPI Challenge datasets. 

The benefit of SynBPS is the full transparency of the data generating process, which can help further understand the effects of different process characteristics on predictive performance. 

Getting Started
================
You can install SynBPS using pip::

    pip install SynBPS

SynBPS requires python 3.10 or higher.

Citation
-----------------

If you use SynBPS, please cite the corresponding paper. The paper can be cited as:
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
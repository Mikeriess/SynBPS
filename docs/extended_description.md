
## How?
The intended usage of this framework is to benchmark new methods within predictive process monitoring. Rather than calibrating a simulation model from an existing process (as in existing frameworks), the aim is to simulate theoretical processes with varying degrees of noise in duration distributions and entropy in the control-flow. The framework uses algorithms (described in the publication), Higher-order Markov Chains (HOMC) and the Hypo-exponential distribution to represent temporal dependency (or its absence) in conditional duration distributions. 

![image](https://github.com/Mikeriess/SynBPS/blob/main/docs/illustration.png)


## Why?
The benefit of SynBPS is in the transparency (and simplicity) of the data generating process, which can help further understand the influence of process characteristics on predictive performance. By e.g. changing the entropy of the process, SynBPS lets you compare the difference in predictive performance across everything between predictable to completely chaotic processes.

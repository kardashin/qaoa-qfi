# QAOA and QFI

This repository contains the programs for our preprint [On the role of overparametrization in Quantum Approximate Optimization
](https://arxiv.org/abs/2508.10086)

## Programs

`QAOA-QFI-minimal.ipynb` contains a minimal example of our QAOA code applied to solving MAX-CUT and MAX-k-SAT problems, as well the code for calculating the quantum Fisher information matrix (rank) for the corresponding ansätze. 

<br>

`cut.py` is a program for solving a random instance of the MAX-CUT problem.
Run the program with the command <br> 
```bash
python cut.py n prob p_min p_max s r path
```
where 
`n` is the number of graph vertices (qubits), `prob` is the edge probability, `p_min` and `p_max` are the minimum and maximum number of ansatz layers, `s` is the seed for the graph generation, `r` is the seed for the initial ansatz parameters generation, and `path` is the path for outputing the raw data.

<br>

`cut_ring.py` is a program for solving the MAX-CUT problem on a ring graph.
Run the program with the command <br> 
```bash
python cut_ring.py n p_min p_max r path
```

<br>

`sat.py` is a program for solving a random instance of the MAX-k-SAT problem.
Run the program with the command <br> 
```bash
python sat.py n k m p_min p_max s r path
```
where 
`n` is the number variables (qubits), `k` is the number of literals in each clause, `m` is the number of clauses, `p_min` and `p_max` are the minimum and maximum number of ansatz layers, `s` is the seed for the SAT instance generation, `r` is the seed for the initial ansatz parameters generation, and `path` is the path for outputing the raw data.

## Data

The folder `Data` contains the *processed* raw data saved into files with the extension `.dict`.
These can be read by the Jupyter Notebooks in the `Results` folder.

In `Data/Raw`, there are Jupyter Notebooks for processing the raw data produced by the programs described above.

Inter alia, the raw data files contain the cost function values at each optimization iteration, and the optimized ansatz angles for each $p$.
To save space, these are not saved into the `.dict` files.
If you need these data, please write to me!

## Plots

The folder `Results` contains the Jupyter Notebooks which process the data in the `Data` folder.
These notebooks can be used to reproduce the plots from the preprint.

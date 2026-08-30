# REGULARIZED DYNAMICAL PARAMETRIC APPROXIMATION OF BOUNDARY VALUE PROBLEMS

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

This repository contains the code to reproduce the figures in the paper:
**"REGULARIZED DYNAMICAL PARAMETRIC APPROXIMATION OF BOUNDARY VALUE PROBLEMS"** by CHRISTOPHE LUIS, PASQUALE NETTIS AND JÖRG NICK.

## Environment & Prerequisites

The computational heavy lifting, including gradient and Jacobian calculations, is powered by JAX. 

To run the codebase, you will need the following primary libraries:
* **JAX / JAX Numpy** 
* **NumPy & SciPy:** 

## Project Structure

The code for this paper is divided into three distinct folders, mapping to the numerical experiments in the text:

1. **`/heat_square/`**: Codebase for the 2D heat equation on a standard square domain (Section 5.2.1).
2. **`/heat_l_shape/`**: Codebase for the 2D heat equation on an L-shaped domain with reentrant corners (Section 5.2.2).
3. **`/schrodinger/`**: Codebase for the Schrödinger equation experiments (Table 1, Section 5.2.3).

## Paper Artifact Mapping

The table below lists the figures from the paper and the exact scripts required to generate them. 

| Paper Artifact | Branch | Computation Script | Plotting Script | Directory |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 5.1** | Main Branch | `num_sol_and_error_plot.py` | `plotting.py` | `/heat_square/` |
| **Figure 5.2** | Main Branch | `main.py` | `convergence_plot.py` | `/heat_l_shape/` |
| **Figure 5.3** | L-shaped Domain Branch | - | `script_plotting_L/conv_an.py` | `/heat_l_shape/` |
| **Figure 5.4** | L-shaped Domain Branch | - | `init_final_error_plot.py` | `/heat_l_shape/` |
| **Figure 5.5** | L-shaped Domain Branch | - | `script_plotting_L/plot_sv.py` | `/heat_l_shape/` |
# Data-driven-Dynamics-Understanding

This repository is the ongoing discovery of system identification, on the path for complete data-driven solutions of systems understanding.

## Source

Code regarding additional implementations required.

- NARMAX:
  - cases_tests.py: Test cases for the system identification of the Barabasi-Albert model with the NARMAX method. Tests are defined and run, results can be found on reports.
  - utils.py: Helper functions for using the SysIdentPy.
    - narmax_state_space: Use NARMAX models for MISO systems.
    - display_nx_model: Display the results SysIdentPy's implementation of the NARMAX model in a readable way regarding the states of a model.
    - solution_to_regressors: Convert a solution array with certain feature names into the regressors matrix structure of SysIdentPy.
- Networks: Source files for the interconnected networks generators.
  - full_degree_matrix.py: File needed to get the degree evolution of all the nodes in a network. This file was replaced with a network generator that returns the degree evolution as well.
  - generator.py: Network generators and supporting functions.
    - _random_subset: Return m unique elements from a sequence.
    - _random_subset_weighted: Return m unique elements from a sequence, with a weight for the elements in the sequence.
    - ba_graph_degree: Returns a random graph using Barabási–Albert preferential attachment, and the degree evolution of the initial nodes with color labels.
    - mixed_graph_degree: Returns a random graph using randomly the rules of Barabási–Albert preferential attachment or simple attachment.
    - ba_fitness_degree: Returns a random graph using Bianconi-Barabási preferential attachment, and the degree evolution of the initial nodes with color labels.
    - ba_discrete_fitness_degree: Returns a random graph using Bianconi-Barabási preferential attachment with predefined fitness values, and the degree evolution of the initial nodes with color labels.
  - nets_degree_generator.py: Multiprocessing network generator; this helps generate an ensemble average of the degree evolution of a network.
  - speed_generators_test.py: Time comparison of different network generators.
  - utils.py: Helper functions for networks analysis.
- SINDy:
  - cases_tests.py: Test cases for the system identification of the Barabasi-Albert model with the SINDy method. Tests are defined and run, results can be found on reports.
- metrics.py: Metrics helper functions.
  - get_metrics_df: Generates a metrics Data-Frame that compares results of the NARMAX and SINDy methods. The metrics used are described in the undergraduate thesis DATA-DRIVEN SYSTEM IDENTIFICATION OF THE BARABASI–ALBERT MODEL.

## Notebooks

Jupyter notebooks that go over different, techniques, simulations, analisys, etc.

- barabasi.ipynb: Explorer of the evolution of the Barabasi-Albert Model
- barabasi_model.ipynb: Comparison of the SINDy and NARMAX methods on the identification of the degree evolution for the node 0 in a Barabasi-Albert Model of 10k nodes with 2 new links per node, in the case of an average ensemble of 100 networks and on a single realisation.
- barabasi_tight_model.ipynb: Comparison of the SINDy and NARMAX methods on the identification of the degree evolution for the node 0in a Bianconi-Barabasi Model of 10k nodes with 2 new links per node and [0.53125, 0.46875] for fitness levels, in the case of an average ensemble of 100 networks and on a single realisation, considering both fitness levels for the initial node.
- barabasi_wide_model.ipynb: Comparison of the SINDy and NARMAX methods on the identification of the degree evolution for the node 0in a Bianconi-Barabasi Model of 10k nodes with 2 new links per node and [0.75, 0.25] for fitness levels, in the case of an average ensemble of 100 networks and on a single realisation, considering both fitness levels for the initial node.
- ode_systems.ipynb: Comparison of the SINDy and NARMAX methods on 5 ODE Systems.
- parameterized_dyn_systems.ipynb: Exploratory comparison of SINDy and NARMAX on a logistic map.
- scir.ipynb: Exploratory comparison of SINDy and NARMAX on the identification of governing equations from pandemic cases data, considering the SCIR, model as a baseline.
- tf_control_model.ipynb: Comparison of the SINDy and NARMAX methods on SISO and MISO transfer functions, with random input.

## Reports

Figures and notes that supported the work (Needs organising and cleaning)

- Figures: All the figures created for analysis or visualisation.
  - Barabasi: Figures results of all the test cases generated for both the SINDy and NARMAX methods.
- Notes: Notes for tests results on model identification and execution times.

## Versions

- 0.0.1:
    - Implemented work regarding the undergraduate thesis: DATA-DRIVEN SYSTEM IDENTIFICATION OF THE BARABASI–ALBERT MODEL
# Scripts directory

This folder contains the end-user scripts for creating and testing the graph neural networks.
To run them, activate first the conda environment via `conda activate env_name`. For checking the arguments needed for running the scripts, type `python name_of_the_script -h`.

- `train_GNN.py` : Run GNN learning process with the desired hyperparameter settings defined in a toml input file. You will find a demo video in the `media` directory.
- `hypopt_GNN.py`: Run Hyperparameter optimization with RayTune using the ASHA algorithm (requires a fix, sorry for that, will be solved asap)
- `GNNvsDFT.py`: Compare GNN prediction to DFT sample you provide.
- `interactive_graph_creator.py`: GNN interface for drawing the graphs and get automatically the related energy and adsorption energy. You will find a demo video in the `media` directory.


# Contributing to the GNN project

First, a big thank you for taking the time to contribute!

# Folder Structure

The structure of the repo is still work in progress, all the Python scripts, modules and interactive notebooks are in the root directory. 

## Subfolders

- `Data`: Stores the VASP training data (named "FG-dataset" in the article), grouped by chemical family in different sub-folders (e.g., `oximes`). Each sub-folder contains at least these two files:  
    1. `structures`: Folder with the CONTCAR geometry files of that specific chemical family, named with the convention `xx-1234-a.contcar`, where `xx` refers to the metal symbol in lowercase, each digit in `123` refers to the number of C, H, O (default, if not S, N, or combinations) atoms in the adsorbate, while `4` is used to distinguish isomers. The last character refers to the adsorption configuration. Example: ethylene on platinum is `pt-2401-a`.
    2. `energies.dat`: A file containing the ground state energy in eV of each sample. Each line represents a sample. Example: `ag-37X3-a -193.89207256`
- `Models`: Contains the Graph Neural Networks that are created or are already created by us. All models are contained in specific folders created after the learning process with the script `train_GNN.py`. Each folder should contain the relevant information about the model.
- `Hyperparameter_Optimization`: This folder is used to store the output of the hyperparameter optimization runs done via `hypopt_GNN.py`.  
- `BM_dataset`: Contains the Big Molecules DFT data used in this work to test the Graph Neural Networks in extrapolation phase.

## Modules

- `constants.py`: Stores the global constants of the GNN framework.
- `functions.py`: General container of functions for data conversion, splitting, target scaling, etc.
- `nets.py`: Contains Graph Neural Network architectures base classes.
- `graph_tools.py`: Functions for manipulating graphs in `torch_geometric.data.Data` format.
- `graph_filters.py`: Filters for graphs in `torch_geometric.data.Data` format.
- `classes.py`: Contains the Dataset class suitable for PyTorch Geometric.
- `post_training.py`: Collecting information of the learning processes and create reports.
- `plot_functions.py`: Data visualization purposes.

## Scripts

- `create_graph_dataset.py`: Convert DFT data to graphs
- `train_GNN.py` : Run GNN learning process with custom hyperparameter setting
- `hypopt_GNN.py`: Run Hyperparameter optimization with RayTune.

## Notebooks

## How to get the DFT data and the models

The DFT data are available in the ioChem-BD repository at the following link. However, here you will find the data not in a structure suitable for the framework, it is just for checking the validity of the samples from DFT point of view. The best option is to download directly the folder `Data` from the Zenodo repository at the following link.

## How to convert the DFT data to Graphs

For each DFT data, two output files are needed to convert it to a graph: the CONTCAR file containing the optimized geometry and the OUTCAR file containing the related ground state energy. 


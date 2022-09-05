# Adsorption Energy of Any Molecule on Metals Using Graph Neural Networks

# Framework installation

1. Clone the repo from GitLab. Open a terminal and use the following command:  

    `git clone https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads.git`  
    
You should have now a folder in the current directory called ``gnn_eads``.  

2. Create a conda environment. Enter the repo: You should find a file called `GNN_env.txt`. It contains all the information about the packages needed to create the environment for the GNN framework. Choose a name for your new environment (we will use `GNN` here) and use the conda command:  

    `conda create --name GNN --file GNN_env.txt`  
    
Check that you have created your new environment by typing `conda env list`: A list with all your environments appear, together with the newly created `GNN`. Use `conda activate GNN` to activate it (you will see the name of the current active environment within parentheses on the left of the terminal).  

3. Install pyRDTP, a Python package used to build the base of the GNN framework. Clone the repo in the existing `gnn_eads` base directory (i.e., be sure to be in `gnn_eads` before executing the following command):  

    `git clone --branch experimental https://gitlab.com/iciq-tcc/nlopez-group/pyrdtp.git`  
    
A new folder `pyrdtp` should be now present in your base repo `gnn_eads`. With the `GNN` environment activated, use pip to install pyRDTP:  

    `pip install pyrdtp`  
    
Done! In theory now everything is set up to start playing with the GNN framework!  

# How to play with the framework

You have two possible choices:

- **Inference mode**: Most Likely, you are a curious person and want to check the performance of the GNN models compared to your DFT systems. In this case, you will test the pre-trained models proposed by us and will need to use a few scripts to run the models. 

- **Training mode**: You will test the whole workflow we had to follow during the creation of the GNN framework. In this case, you will also need the DFT datasets that we used to train the models. Within this mode, you can train GNN models with your own hyperparameter setting, or, if you have enough computational resources, perform hyperparameter optimization with our workflow based on RayTune.

## Inference mode

Within this mode, you can opt among two different options:

1. You already have some DFT calculations and want to compare the performance of the GNN models with the ground truth from DFT already available to you. In this case, the main scripts and files you will work with are...

2. You have no DFT calculations for a specific system and want to get a direct estimation from GNNs. In this case, you will play mainly with `interactive_graph_creator.py`, a Python script that will help you build the graph related to your specific case, automatically providing the ground state energy of the system and the adjusted adsorption energy. In order to get the final adsorption energy, just substract the energy of your metal slab. 
## Training mode

The DFT datasets and the models are stored in Zenodo (link).


# Authors

Santiago Morandi, Ph.D. Student, Lopez group at ICIQ (Spain)  
Sergio Pablo Garcia Carrillo, Ph.D. Student at Lopez Group (ICIQ, Spain), now postdoc in the Aspuru-Guzik group (UoT, Canada)  


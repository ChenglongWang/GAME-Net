# Adsorption Energy of Any Molecule on Metals Using Graph Neural Networks

# Framework installation

1. Clone the repo from GitLab. Open a terminal from the desktop and use the following command:  
    `git clone https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads.git`  
You should have now a folder in the Desktop called ``gnn_eads``.  
2. Create conda environment. Enter the repo: You should find a text file called `GNN_env.txt`. Choose a name for your new environment (e.g., `GNN` here) and use the following conda command:  
    `conda create --name GNN --file GNN_env.txt`  
Check that you have created your new new conda envionment by typing `conda env list`: A list with all your environments should appear, together with the newly created `GNN`. Use `conda activate GNN` to activate it.  
3. Install pyRDTP, a Python package used to build the base of the GNN framework. Clone the repo in the existing `gnn_eads` base directory:  
    `git clone --branch experimental https://gitlab.com/iciq-tcc/nlopez-group/pyrdtp.git`  
A new folder `pyrdtp` should be present in your base repo. Now, with the `GNN` environment activated, install pyRDTP with pip:  
    `pip install pyrdtp`  
4. Done, in theory now eveything is set up in order to start playing with the GNN framework!  

# Authors

Santiago Morandi, Ph.D. Student, Lopez group at ICIQ (Spain)  
Sergio Pablo Garcia Carrillo, Ph.D Student at Lopez Group (ICIQ, Spain), now postdoc in the Aspuru-Guzik group (UoT, Canada)  


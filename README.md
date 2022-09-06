# Adsorption Energy of Any Molecule on Metals Using Graph Neural Networks

<p align="center">
    <img src="./GNN.gif" width="60%" height="60%"/>
</p>

This is the official repository of the framework related to the article "Adsorption Energy of Any Molecule on Metals Using Graph Neural Networks". The Graph Neural Networks (GNNs) developed within this framework allow the fast prediction of the DFT ground state energy of the following systems:

- All Gas-phase closed-shell molecules containing C, H, O, N and S.
- Adsorption systems: Same molecules mentioned above on the following 14 metals: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn.

The framework is built on top of [PyTorch](https://www.pytorch.org/) and [PyTorch Geometric](https://www.pytorch-geometric.readthedocs.io/en/latest/index.html).
## Installation

1. Clone the repo from GitLab. Open a terminal and use the following command:  

    `git clone https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads.git`  
    
    You should now have the repo ``gnn_eads`` in your current directory.  

2. Create a conda environment. Enter the repo: You should find the file `GNN_env.txt`: It contains all the information about the packages needed to create the environment for the GNN framework. Choose a name for the new environment (we will use `GNN` here) and type the command:  

    `conda create --name GNN --file GNN_env.txt`  
    
    Check that you have created your new environment by typing `conda env list`: A list with all your environments appear, together with the newly created `GNN`. Activate it with `conda activate GNN` (you will see the name of the current active environment within parentheses on the left of the terminal).  

3. Install pyRDTP, a package at the base of the GNN framework. Clone the repo in the existing `gnn_eads` base directory (i.e., be sure to be in `gnn_eads` before executing the following command) and install it with pip:  

    `git clone --branch experimental https://gitlab.com/iciq-tcc/nlopez-group/pyrdtp.git`  
    `pip install pyrdtp`

    To check the correctness of the installation, use `conda list` and check out the presence of pyrdtp in the list.

4. Create the following empty folders in the repo: 

    `mkdir Data Models Hyperparameter_Optimization`

    As the names suggest, `Data` will store the DFT datasets used to train the GNNs in case you will work in *training mode* (see below), while `Models` will contain the different GNN models stored in [Zenodo](https://www.zenodo.org/) if you choose to work in *inference mode* (see below), or the models you will create if you work in training mode otherwise. `Hyperparameter_Optimization` will store the results of the tuning process.

Done! In theory now everything is set up to start playing with the GNN framework!

## Usage

You have two possible choices for interacting with the GNN framework:

- **Inference mode**: Most likely, you are a curious person and want to probe the performance of the GNN models compared to your DFT simulations. In this case, you will test the pre-trained models proposed by us without going deeper in the details behind the models generation and without accessing the DFT data used to train them. 

- **Training mode**: You will test the whole workflow we had to follow during the framework development. In this case, you will need the DFT datasets for training the GNN. Within this mode, you can train your own models with the preferred hyperparameter setting and model architecture, or, if you have enough computational resources, you can also perform hyperparameter tuning with the workflow based on Ray Tune.

### Inference mode

Within this mode, you can opt among two different options:

1. You already have performed some DFT calculations with VASP and want to compare the performance of the GNN models with the ground-truth provided by your data. In this case, the main scripts and files you will work with are `GNNvsDFT.ipynb` and `GNNvsDFT.ipynb`

2. You have no DFT data for a specific system and want to get an estimation from our pre-trained Graph Neural Networks. In this case, you will play mainly with `interactive_graph_creator.py`, an interactive script connected to the [ChemSpider](https://www.chemspider.com) database to help you build the graph related to your specific case, automatically providing the ground state energy of the system and the adjusted adsorption energy. 
### Training mode

The DFT datasets and the models are stored open-access [here](). The DFT datasets are also uploaded to ioChem-BD [doi]().
Within this mode as well, you can choose among two available ways to work with the GNN:

1. Perform a model training with your own model architecture and hyperparameter setting: To do so, follow the instructions provided in the Jupyter notebook `train_GNN.ipynb`, or directly run the script `train_GNN.py`.

2. Perform a hyperparameter optimization using the Asynchronous Successive HAlving (ASHA) scheduler provided by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). You can study the effect of all hyperparameters on the final model performance (e.g., learning rate, loss function, epochs) and you can also test different model architecture automatically, without the need to manually define the model architecture. the script you have to use in this case is `hypopt_GNN.py`.


# Authors

Santiago Morandi, Ph.D. Student, Lopez group at ICIQ (Spain)  
Sergio Pablo Garcia Carrillo, Ph.D. Student at Lopez Group (ICIQ, Spain), now postdoc in the Aspuru-Guzik group (UoT, Canada)
# Contributors

Zarko Ivkovic, B.Sc. Student, DFT dataset creation and interface testing.

# License

MIT

# Support 

In case you need help or are interested in contributing, feel free to contact us sending an e-mail to smorandi@iciq.es


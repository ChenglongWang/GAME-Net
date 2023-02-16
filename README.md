# Fast Evaluation of the Adsorption Energy of Organic Molecules on Metals via Graph Neural Networks

<p align="center">
    <img src="./media/GNN2.gif" width="60%" height=60%"/>
</p>

## Overview 

This is the repository of the framework related to the work "Fast Evaluation of the Adsorption Energy of Organic Molecules on Metals via Graph Neural Networks", preprint [here](https://chemrxiv.org/engage/chemrxiv/article-details/633dbc93fee74e8fdd56e15f), where we introduce GAME-Net (*Graph-based Adsorption on Metal Energy-neural Network*), a graph neural network developed for the fast prediction of the DFT ground state energy of the following systems:

- Closed-shell molecules containing C, H, O, N and S.
- Mentioned molecules adsorbed on 14 transition metals: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, and Zn.

The framework and related model have been built with [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://www.pyg.org/) and [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

## Installation (Linux systems)

Prerequisites for installing the Python code repository are [git](https://git-scm.com/) and [conda](https://docs.conda.io/en/latest/).

1. Clone the repo from GitLab. Open a terminal and type the following command:  

    `git clone https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads.git`  
    
    You should now have the repo ``gnn_eads`` in your current directory.  

2. Create a conda environment. Enter the repo. You should find the file `requirements.txt`: It contains the information about the packages needed to create the environment for this project (NB: The environment occupies 7 GB, check that you have enough space before installing it).   

    `conda create --name GNN --file requirements.txt`  
    
    Check that you have created the new environment by typing `conda env list`: A list with all your environments appears, together with the newly created `GNN`. Activate it with `conda activate GNN` (you will see the name of the current active environment within parentheses on the left of the terminal prompt).  

3. Install [PyRDTP](https://gitlab.com/iciq-tcc/nlopez-group/pyrdtp), an in-house package for manipulating chemical structures, and [Ray](https://docs.ray.io/en/latest/index.html), a tool for performing hyperparameter optimization studies. Since they are not available in the conda channels, use pip to install them:  
  
    `pip install pyrdtp ray`

    To check the correctness of the installation, type `conda list` and check out the presence of pyrdtp and ray in the list.

4. (NEW!) To use GAME-Net locally as a web application, install Django and pydot:

    `conda install -c conda-forge pydot`
    `conda install -c anaconda django`
    `pip install django-cors-headers`

Done! Now everything is set up to start playing with the GNN framework!

## Usage

You have two possible modes:

- **Inference mode**: Most likely, you are a curious person and want to probe the performance of the GNN models compared to your DFT simulations. In this case, you will test the models developed by us, without going deeper in the details behind the models' creation process and without accessing the DFT data used to train them. 

- **Training mode**: You will go through all the steps defined in the workflow for the model generation process. In this case you will need the raw DFT datasets for training the GNN. Within this mode, you can train your own models with the preferred hyperparameter setting and model architecture, or, if you have enough computational resources, you can perform hyperparameter tuning with the workflow based on [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). 

### Inference mode

Within this mode, you can opt between three different options:

1. You already performed some DFT calculations with [VASP](https://www.vasp.at/) and want to compare the performance of the GNN models with the ground-truth provided by your data. In this case, the main script you will work with is `GNNvsDFT.py`.

2. You have no DFT data for a specific system and want to get an estimation from our trained Graph Neural Networks. In this case, you will play with `interactive_graph_creator.py`, a GNN interface connected to the [ChemSpider](https://www.chemspider.com) database to help you draw the graph related to your specific case, providing the ground state energy of the system and the adsorption energy. See the demos in the `Media` directory. 

    `python interactive_graph_creator.py`

3. You can use GAME-Net as a web application. To run, type the following script from the repo root:

    `python web/manage.py runserver --insecure`

### Training mode

The DFT datasets are stored in [ioChem-BD](https://doi.org/10.19061/iochem-bd-1-257), and the needed samples for GAME-Net are in the `data/FG_dataset` folder.
Within this mode, you can choose among three available ways to use the GNN:

1. Perform a model training with your own model architecture and hyperparameter setting: To do so, follow the instructions provided in the Jupyter notebook `train_GNN.ipynb`, or directly run the script `train_GNN.py`. The hyperparameter settings must be provided via a .toml file (you will find some input templates in the `scripts` folder). Once created, type: 

    `python train_GNN.py -i hyper_config.toml`

    To check the documentation of the script, type `python train_GNN.py -h`.

2. Run a nested cross-validation to assess the generalization performance on the FG-dataset. The script for this task is `nested_cross_validation_GNN.py`.

3. Perform a hyperparameter optimization using the Asynchronous Successive Halving (ASHA) scheduler provided by Ray Tune. You can study the effect of all hyperparameters on the final model performance (e.g., learning rate, loss function, epochs) and you can also test different model architecture automatically, without the need of manually defining the architecture. The script you have to use in this case is `hypopt_GNN.py`. For this script, the hyperparameter space must be defined in the script before running it. For example, to launch a hyperparameter optimization called `hypopt_test` with 2000 trials, each one with a grace period of 15 epochs and providing 0.5 GPUs for each trial (e.g., two trials per GPU), type:

    `python hypopt_GNN.py -o hypopt_test -s 2000 -gr 15 -gpt 0.5`


## Notebooks

To reproduce some of the results presented in the article, we provide some Jupyter notebooks in the `notebooks` folder. These contain information mainly related to GAME-Net usage with different datasets (BM-dataset, external literature datasets).

# Authors

Santiago Morandi, doctoral researcher, Núria López group (Institut Català d'Investigació Química, Spain)  
Sergio Pablo-García, postdoctoral researcher, The Matter Lab, Alán Aspuru-Guzik group (University of Toronto, Canada)

# Contributors

Žarko Ivković, M.Sc. student, University of Barcelona, ICIQ Summer Fellow 2022; involved in the creation of part of the DFT dataset and interface testing.

# License

[MIT License](https://gitlab.com/iciq-tcc/nlopez-group/gnn_eads/-/blob/master/LICENSE)

# Support 

In case you need help or are interested in contributing, feel free to contact us sending an e-mail to smorandi@iciq.es

# Acknowledgements

<p align="center">
    <img src="./media/ack_repo.png"/>
</p>
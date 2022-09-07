"""Python script for the direct comparison of the GNN performances compared to the DFT ground-truth."""

__author__ = "Santiago Morandi"

import argparse

import torch

from functions import get_graph_sample, get_mean_std_from_model
from graph_tools import plotter

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT calculation to graph and compare the GNN performance to the DFT ground truth")
    PARSER.add_argument("-m", "--model", type=str, dest="model", 
                        help="Name of the GNN model.")
    PARSER.add_argument("-i", "--input", type=str, dest="input", 
                        help="Path to the DFT folder of the specific adsorption system/gas-molecule.")
    PARSER.add_argument("-s", "--slab", type=str, default=None, dest="slab", 
                        help="Path to the DFT folder containing the empty slab in case of adsorption system.")
    PARSER.add_argument("-f", "--family", type=str, default=None, dest="family", 
                        help="Tag for the graph object")
    PARSER.add_argument("-sm", "--surface-multiplier", type=int, default=None, dest="sm", 
                        help="Surface multiplier in the case the slab is an extension of the empty slab given as input.")
    
    ARGS = PARSER.parse_args()
    
    # 1) Load pre-trained GNN model 
    MODEL_NAME = ARGS.model
    MODEL_PATH = "./Models/{}/".format(MODEL_NAME)    
    model = torch.load("{}/model.pth".format(MODEL_PATH))
    model.load_state_dict(torch.load("{}GNN.pth".format(MODEL_PATH)))
    model.eval()
    model.to("cpu")  
    mean_tv, std_tv = get_mean_std_from_model(MODEL_PATH)
    
    # 2) Convert input DFT sample to graph object
    graph = get_graph_sample(ARGS.input, ARGS.slab, family=ARGS.family, surf_multiplier=ARGS.sm)
    print(graph)
    
    # 3) Get prediction from GNN model
    E_GNN = model(graph).item() * std_tv + mean_tv
    abs_error = abs(E_GNN - graph.y)
    result = "{}: GNN = {:.2f} eV    DFT = {:.2f} eV    abs.err. = {:.2f} eV".format(graph.formula.strip(), 
                                                                                     E_GNN, 
                                                                                     graph.y, 
                                                                                     abs_error)
    print(result)
    plotter(graph)
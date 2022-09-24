"""Python script for the direct comparison of the GNN performance compared to the DFT."""

__author__ = "Santiago Morandi"

import argparse

from functions import get_graph_sample
from graph_tools import plotter
from nets import PreTrainedModel

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT system to graph and compare the GNN prediction to the DFT outcome.")
    PARSER.add_argument("-m", "--model", type=str, dest="model", 
                        help="Name of the GNN model (must be present in the Models folder).")
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
    model = PreTrainedModel("./Models/{}/".format(ARGS.model))
    
    # 2) Convert input DFT sample to graph object
    graph = get_graph_sample(ARGS.input, ARGS.slab,
                             model.g_tol,
                             model.g_sf,
                             model.g_metal_2nn,
                             family=ARGS.family,
                             surf_multiplier=ARGS.sm)
    print(graph)
    
    # 3) Get GNN prediction
    E_GNN = model.evaluate(graph)
    abs_error = abs(E_GNN - graph.y)
    result = "{}: GNN = {:.2f} eV    DFT = {:.2f} eV    abs.err. = {:.2f} eV".format(graph.formula.strip(), 
                                                                                     E_GNN, 
                                                                                     graph.y, 
                                                                                     abs_error)
    print(result)
    plotter(graph)
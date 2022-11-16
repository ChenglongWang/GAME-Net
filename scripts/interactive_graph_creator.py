"""Interactive script for predicting the adsorption energy of molecules on metals with Graph Neural Networks"""

import time
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/gnn_eads')))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from chemspipy import ChemSpider
from pyRDTP.tools.pymol import from_smiles

from gnn_eads.constants import METALS, ENCODER, CORDERO, MOL_ELEM, CHEMSPIPY
from gnn_eads.functions import connectivity_search_voronoi, ensemble_to_graph, surf
from gnn_eads.nets import PreTrainedModel
from gnn_eads.graph_tools import visualize_graph

           
cs = ChemSpider(CHEMSPIPY)
to_int = lambda x: [float(i) for i in x]
    
# 1) Load model
MODEL_PATH = "../models/best_model"    
model = PreTrainedModel(MODEL_PATH)

#2) Interactive section
print("----------------------------------------------------")
print("Welcome to the graph neural network (GNN) interface!")
print("----------------------------------------------------")
adsorbate = input("1) Type the name of the molecule (ex. ethylene oxide): ")
result = cs.search(adsorbate)[0]
SMILES = result.smiles
formula = result.molecular_formula
print("Molecule: {}    Formula: {}    SMILES: {}\n".format(adsorbate, formula, SMILES))
molecule = from_smiles(SMILES)
elem_rad = {}  # Atomic radius dict for graph edge detection
for metal in METALS:
    elem_rad[metal] = CORDERO[metal] * model.g_sf
for element in MOL_ELEM:
    elem_rad[element] = CORDERO[element]
molecule = connectivity_search_voronoi(molecule, model.g_tol, elem_rad)
graph = ensemble_to_graph(molecule, model.g_metal_2nn)
elem, source, target = list(graph[1][0]), list(graph[1][1][0]), list(graph[1][1][1])
adsorption = input("2) Do you want to evaluate the molecule on an adsorption configuration?[y/n]: ")
if adsorption == "n":
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = ENCODER.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    gas_graph = Data(x=x, edge_index=edge_index)
    visualize_graph(gas_graph, font_color="black")
    plt.show()
    time0 = time.time()
    gnn_energy = model.evaluate(gas_graph)
    gnn_time = time.time() - time0
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("System: {} (gas phase)".format(adsorbate.capitalize()))
    print("Molecule energy = {:.2f} eV (PBE + VdW)".format(gnn_energy))
    print("Execution time = {:.2f} ms".format(gnn_time * 1000.0))
elif adsorption == "y":
    elem_array_ads = np.array(elem).reshape(-1, 1)
    elem_enc_ads = ENCODER.transform(elem_array_ads).toarray()
    edge_index_ads = torch.tensor([source, target], dtype=torch.long)
    x_ads = torch.tensor(elem_enc_ads, dtype=torch.float)
    adsorbate_graph = Data(x=x_ads, edge_index=edge_index_ads)    
    metal = None
    while metal not in METALS:
        metal = input("3) Define metal species (e.g., Pd): ").capitalize()
        if metal not in METALS:
            print("Available metals are Ag, Au, Cd, Cu, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn.")
    n_metals = int(input("4) Define the number of metal atoms interacting with the adsorbate: "))
    moldict = dict(zip(list(range(len(elem))), elem))
    print("Legend: {}".format(moldict))
    for i in range(n_metals):
        elem.append(metal)
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = ENCODER.transform(elem_array).toarray()
    x = torch.tensor(elem_enc, dtype=torch.float)
    for metal_atom in range(n_metals):
        edges_per_metal = int(input("{} atom {}: Define number of connections: ".format(metal, metal_atom+1)))
        for bond in range(edges_per_metal):
            source.append(metal_atom + len(elem) - n_metals)
            y = int(input("{} atom {}, connection {}: Define index of the connected element: ".format(metal, metal_atom+1, bond+1)))
            if y >= len(elem):
                raise ValueError("Wrong defined connection. Check the printed legend")
            target.append(y)
            source.append(y)
            target.append(metal_atom + len(elem) - n_metals)
    edge_index = torch.tensor([source, target], dtype=torch.long)
    ads_graph = Data(x=x, edge_index=edge_index)
    visualize_graph(ads_graph, font_color="black")
    plt.show()
    time0 = time.time()
    E_adsorbate = model.evaluate(adsorbate_graph)
    E_ensemble = model.evaluate(ads_graph)
    E_adsorption = E_ensemble - E_adsorbate
    gnn_time = time.time() - time0
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("System: {} on {}-({})".format(adsorbate.capitalize(), metal.capitalize(), surf(metal.capitalize())))
    print("Ensemble energy = {:.2f} eV (PBE + VdW)".format(E_ensemble))
    print("Molecule energy = {:.2f} eV (PBE + VdW)".format(E_adsorbate))
    print("Adsorption energy = {:.2f} eV".format(E_adsorption))
    print("Execution time = {:.2f} ms".format(gnn_time *1000.0))         
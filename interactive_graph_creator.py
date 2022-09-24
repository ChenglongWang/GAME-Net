"""Interactive script for predicting the adsorption energy of molecules on metals with Graph Neural Networks"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from chemspipy import ChemSpider
from pyRDTP.tools.pymol import from_smiles

from constants import METALS, ENCODER, CORDERO, MOL_ELEM
from functions import connectivity_search_voronoi, ensemble_to_graph
from nets import PreTrainedModel
from graph_tools import plotter

MODEL = "LONG_NIGHT_full"            
API_KEY = "NmdJjTAaDxkBoDYGrSRLQED9zKOhmqJ9"  # Chemspider key to access database
cs = ChemSpider(API_KEY)
to_int = lambda x: [float(i) for i in x]

# 1) Load model
model = PreTrainedModel(MODEL)
# 2) Interactive section
print("---------------------------------------------------")
print("Welcome to the graph neural network (GNN)interface!")
print("---------------------------------------------------")
print("Author: Santiago Morandi (ICIQ)\n")
adsorbate = input("1) Type the name of the molecule (ex. Vitamin C): ")
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
    plotter(gas_graph)
    plt.show()
    gnn_energy = model.evaluate(gas_graph)
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("GNN energy = {:.2f} eV (PBE + VdW)".format(gnn_energy))
elif adsorption == "y":
    elem_array_ads = np.array(elem).reshape(-1, 1)
    elem_enc_ads = ENCODER.transform(elem_array_ads).toarray()
    edge_index_ads = torch.tensor([source, target], dtype=torch.long)
    x_ads = torch.tensor(elem_enc_ads, dtype=torch.float)
    adsorbate = Data(x=x_ads, edge_index=edge_index_ads)    
    metal = None
    while metal not in METALS:
        metal = input("3) Define metal species (e.g., Pd): ")
        if metal not in METALS:
            print("Unknown species (Available metals: Ag Au Cd Cu Ir Ni Os Pd Pt Rh Ru Zn)")
    n_metals = int(input("4) Define the number of metal atoms interacting with the adsorbate: "))
    moldict = dict(zip(list(range(len(elem))), elem))
    print("Legend: {}".format(moldict))
    for i in range(n_metals):
        elem.append(metal)
    for metal_atom in range(n_metals):
        x = int(input("{} atom {}: Define number of connections: ".format(metal, metal_atom+1)))
        for bond in range(x):
            source.append(metal_atom + len(elem))
            y = int(input("{} atom {}, connection {}: define index of the connected element: ".format(metal, metal_atom+1, bond+1)))
            if y >= len(elem):
                raise ValueError("Wrong defined connection. Check the printed legend")
            target.append(y)
            source.append(y)
            target.append(metal_atom + len(elem))
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = ENCODER.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    ads_graph = Data(x=x, edge_index=edge_index)
    plotter(ads_graph)
    plt.show()
    E_ensemble = model.evaluate(ads_graph)
    E_molecule = model.evaluate(adsorbate)
    E_adsorption = E_ensemble - E_molecule
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("GNN ensemble energy = {:.2f} eV (PBE + VdW)".format(E_ensemble))
    print("GNN molecule energy = {:.2f} eV (PBE + VdW)".format(E_molecule))
    print("GNN adsorption energy = {:.2f} eV".format(E_adsorption))
    
    

            
    
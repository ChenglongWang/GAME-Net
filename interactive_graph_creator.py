"""Interactive script for predicting the adsorption energy of molecules on metals via SantyxNet Graph Neural Network"""

from pyRDTP.tools.pymol import from_smiles
from constants import METALS, node_features, encoder
from chemspipy import ChemSpider
from functions import connectivity_search_voronoi, ensemble_to_graph
from torch_geometric.data import Data
import torch
from graph_tools import plotter
from nets import SantyxNet
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# LOAD CHEMSPIDER API KEY
API_KEY = "NmdJjTAaDxkBoDYGrSRLQED9zKOhmqJ9"  
cs = ChemSpider(API_KEY)
to_int = lambda x: [float(i) for i in x]
# LOAD MODEL
MODEL_PATH = "./Models/FG1"
model = SantyxNet(dim=128, node_features=node_features)
model.load_state_dict(torch.load("{}/GNN.pth".format(MODEL_PATH)))
model.eval()
file = open("{}/performance.txt".format(MODEL_PATH))
lines = file.readlines()
mean_tv = float(lines[3].split()[-2])
std_tv = float(lines[4].split()[-2])
# INTERACTIVE SECTION
print("---------------------------------------------------")
print("Welcome to the Graph Generator for the GNN Project!")
print("---------------------------------------------------")
print("Author: Santiago Morandi (ICIQ)\n")
adsorbate = input("1) Type the name of the molecule (ex. carbon dioxide): ")
result = cs.search(adsorbate)[0]
SMILES = result.smiles
formula = result.molecular_formula
print("Molecule: {}    Formula: {}    SMILES: {}\n".format(adsorbate, formula, SMILES))
molecule = from_smiles(SMILES)
molecule = connectivity_search_voronoi(molecule)
graph = ensemble_to_graph(molecule)
elem = list(graph[1][0])
n_mol = len(elem)
source = list(graph[1][1][0])
target = list(graph[1][1][1])
adsorption = input("2) Do you want to evaluate the molecule on an adsorption configuration?[y/n]: ")
if adsorption == "n":
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    plotter(data, figsize=(6,6))
    plt.show()
    DL = DataLoader([data], batch_size=1, shuffle=False)   
    for batch in DL:
        energy = model(batch)
    energy = energy.item() * std_tv + mean_tv
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("Gas energy = {:.2f} eV (PBE + VdW)".format(energy))
elif adsorption == "y":
    elem_array_ads = np.array(elem).reshape(-1, 1)
    elem_enc_ads = encoder.transform(elem_array_ads).toarray()
    edge_index_ads = torch.tensor([source, target], dtype=torch.long)
    x_ads = torch.tensor(elem_enc_ads, dtype=torch.float)
    adsorbate = Data(x=x_ads, edge_index=edge_index_ads)    
    metal = None
    while metal not in METALS:
        metal = input("3) Define metal species (e.g., Pd): ")
        if metal not in METALS:
            print("Unknown species (available metals: Ag Au Cd Cu Ir Ni Os Pd Pt Rh Ru Zn)")
    n_metals = int(input("4) Define the number of metal atoms interacting with the adsorbate: "))
    moldict = dict(zip(list(range(n_mol)), elem))
    print("Legend: {}".format(moldict))
    for i in range(n_metals):
        elem.append(metal)
    for metal_atom in range(n_metals):
        x = int(input("{} atom {}: Define number of connections: ".format(metal, metal_atom+1)))
        for bond in range(x):
            source.append(metal_atom + n_mol)
            y = int(input("{} atom {}, connection {}: define index of the connected element: ".format(metal, metal_atom+1, bond+1)))
            if y >= n_mol:
                raise ValueError("Wrong defined connection. Check the printed legend")
            target.append(y)
            source.append(y)
            target.append(metal_atom + n_mol)
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    plotter(data, figsize=(6,6))
    plt.show()
    DL = DataLoader([data, adsorbate], batch_size=2, shuffle=False)   
    for batch in DL:
        energy = model(batch)
    E_ensemble = energy[0].item() * std_tv + mean_tv
    E_molecule = energy[1].item() * std_tv + mean_tv
    E_adsorption = E_ensemble - E_molecule
    print("-----------------------------------")
    print("-----------GNN PREDICTION----------")
    print("-----------------------------------")
    print("Ensemble energy = {:.2f} eV (PBE + VdW)".format(E_ensemble))
    print("Molecule energy = {:.2f} eV (PBE + VdW)".format(E_molecule))
    print("Adsorption energy = {:.2f} eV".format(E_adsorption))
    
    

            
    
"""Module containing functions for handling and visualization purposes."""

__author__ = "Santiago Morandi"

from pyRDTP.data.colors import rgb_colors
import numpy as np
import networkx
import torch 
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from constants import METALS, MOL_ELEM, NODE_FEATURES
from functions import get_graph_formula

complete_list = METALS + MOL_ELEM
complete_list.sort()

def convert_gpytorch_to_networkx(graph: Data) -> networkx.Graph:
    """
    Convert graph object from pytorch_geometric to networkx type.    
    For each node in the graph, the label corresponding to the atomic species 
    is added as attribute.
    Args:
        graph(torch_geometric.data.Data): torch_geometric graph object.
    Returns:
        new_g(networkx.classes.graph.Graph): networkx graph object.
    """
    node_features_matrix = graph.x.numpy()
    n_nodes = graph.num_nodes
    atom_list = []
    for i in range(n_nodes):
        index = np.where(node_features_matrix[i,:] == 1)[0][0]
        atom_list.append(complete_list[index])
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    connections = list(g.edges)
    new_g = networkx.Graph()
    # Node attributes: chemical element + associated color
    for i in range(n_nodes):
        new_g.add_node(i, atom=atom_list[i], rgb=rgb_colors[atom_list[i]])
    new_g.add_edges_from(connections, minlen=2)
    return new_g

def convert_networkx_to_gpytorch(graph: networkx.Graph) -> Data:
    """
    Convert graph object from networkx to pytorch_geometric type.
    Args:
        graph(networkx.classes.graph.Graph): networkx graph object
    Returns:
        new_g(torch_geometric.data.Data): torch_geometric graph object        
    """
    pass

def plotter(graph,
            node_size: int=400,
            font_color: str="white",
            font_weight: str="bold",
            alpha: float=0.9, 
            arrowsize: int=10,
            width: float=1.2,
            k: float=0.01,
            scale: float=1,
            dpi: int=600,
            figsize: tuple[int, int]=(4,4)):
    """
    Plot graph with atom labels and colors. 
    Kamada_kawai_layout engine is applied as it gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object of the chemical ensemble.
    """
    graph = convert_gpytorch_to_networkx(graph)
    labels = networkx.get_node_attributes(graph, 'atom')
    colors = list(networkx.get_node_attributes(graph, 'rgb').values()) 
    plt.figure(figsize=figsize, dpi=dpi) 
    networkx.draw_networkx(graph, 
                     labels=labels, 
                     node_size=node_size,
                     font_color=font_color, 
                     font_weight=font_weight,
                     node_color=colors, 
                     alpha=alpha, 
                     arrowsize=arrowsize, 
                     width=width,
                     pos=networkx.kamada_kawai_layout(graph))                    
    
def extract_adsorbate(graph: Data) -> Data:
    """Extract molecule from the adsorption configuration graph,
    removing metals and connections between metal and molecule.
    
    Args:
        graph (torch_geometric.data.Data): Adsorption system in graph format
    Returns:
        adsorbate(torch_geometric.data.Data): Adsorbate molecule in graph format
    """    
    CHONS = {2, 5, 7, 9, 15} # indexes of C, H, O, N, S in the encoder
    y = [None] * graph.num_nodes # function for new indexing
    node_list = []  
    node_index = []
    edge_list = []
    edge_index = []
    # 1) Node selection 
    counter = 0
    for atom in range(graph.num_nodes):
        index = torch.where(graph.x[atom, :] == 1)[0].item()
        if index in CHONS:
            y[atom] = counter
            node_index.append(atom)
            node_list.append(graph.x[atom, :])
            counter += 1
    def ff(num):  # new indexing for the new graph (important!)
        return y[num]
    # 2) Edge selection
    for link in range(graph.num_edges):
        nodes = graph.edge_index[:, link]
        switch = 0
        for node in nodes:
            if node not in node_index:
                switch = 1
        if switch == 0:
            edge_list.append(nodes)
            edge_index.append(link)
        switch = 0
    # 3) Graph construction
    x = torch.zeros((len(node_list), NODE_FEATURES))
    edge = torch.zeros((2, len(edge_index)))
    for i in range(x.shape[0]):
        x[i, :] = node_list[i]
    for j in range(2):
        for k in range(edge.shape[1]):
            edge[j, k] = ff(edge_list[k][j])
    adsorbate = torch_geometric.data.Data(x, edge)
    return adsorbate

def get_number_atoms(graph: Data, atom: str) -> int:
    """Return number of atoms of a specific element in the graph.

    Args:
        graph (torch_geometric.data.Data): graph sample
        atom (str): atomic element present in the encoder
    Returns:
        n(int): number of atoms of the specified element in the graph
    """
    formula = get_graph_formula(graph)
    if atom in formula:
        index = formula.find(atom)
        return int(formula[index+1])
    else:
        return "The defined element is not present in the system under study"
    



        
"""
Module containing a set of filter functions for graphs in the Geometric PyTorch format.
These filters are applied before the inclusion of the graphs in the HetGraphDataset objects.
"""
from constants import ENCODER, METALS, ELEMENT_LIST, NODE_FEATURES
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
from graph_tools import *
from functions import get_graph_formula

def single_fragment_filter(graph: torch_geometric.data.Data):
    """
    Graph filter function that checks that there is just one adsorbate on the original system.
    Args: 
        graph(torch_geometric.data.Data): Graph object representation.
    Returns:
        (bool): True = Just one single adsorbate is present in the original graph
                False = More fragments are represented by the graph.
    """
    categories = list(ENCODER.categories_[0])
    switch = []
    # Non-metal nodes are kept
    for i in range(graph.num_nodes):
        index = torch.where(graph.x[i, :] == 1)[0].item()
        if categories[index] in METALS:
            switch.append(1)
        else:
            switch.append(0)
    N = switch.count(0)
    x = torch.zeros((N, len(categories)))
    counter = 0
    func  = [None] * graph.num_nodes
    for i in range(graph.num_nodes):
        if switch[i] == 0:
            x[counter, :] = graph.x[i, :]
            func[i] = counter
            counter += 1             
    # Connections not involving metals are kept
    def ff(number):
        return func[number]
    connections = []
    switch = 0
    for i in range(graph.num_edges):
        connection = list(graph.edge_index[:, i])
        connection = [node.item() for node in connection]
        for j in connection:
            index = torch.where(graph.x[j, :] == 1)[0].item()
            if categories[index] in METALS:
                switch = 1
        if switch ==  0:
            connections.append(connection)
        else: # don't append connections, restart the switch
            switch = 0
    for bond in range(len(connections)):
        connections[bond] = [ff(i) for i in connections[bond]]
    connections = torch.Tensor(connections)
    graph_new = Data(x, connections.t().contiguous())
    graph_nx = to_networkx(graph_new, to_undirected=True)
    nx.is_connected(graph_nx)
    if graph_new.num_nodes != 1 and graph_new.num_edges != 0:
        if nx.is_connected(graph_nx):
            return True
        else:
            print("{} removed: more than one adsorbate!".format(get_graph_formula(graph, ENCODER.categories_[0])))
            return False 
    else:  # graph with one node and zero edges
        return True

def H_connectivity_filter(graph: torch_geometric.data.Data):
    """
    Graph filter function that checks the connectivity of H atoms in the adsorbate is correct.
    Each H atoms must be connected to maximum one atom among C, H, O, N, S in the molecule.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
    Returns:
        (bool): True= Correct H-connectivity in the molecule
                False= Bad H-connectivity in the molecule
    """
    # 1) Find index of H nodes
    H_nodes_indexes = []
    for i in range(graph.num_nodes):
        if graph.x[i, 5] == 1:  # 5 is the index of H in the one-hot encoder
            H_nodes_indexes.append(i)
    # 2) Apply correctness criterion to all H atoms
    # NB Just one bad connectivity makes the whole graph wrong
    bad_H = 0
    CHONS = {2, 5, 7, 9, 15} # indexes of C, H, O, N, S in the encoder
    for index in H_nodes_indexes:
        counter = 0  # edges between H and atoms in (C, H, O, N, S)
        for j in range(graph.num_edges):
            if index == graph.edge_index[0, j]:  # NB: Each edge repeated twice in order to have undirected
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0].item()  # most delicate part!
                if  other_atom in CHONS: 
                    counter += 1
        if counter not in (0, 1): # Exactly one connection with CHONS or just H atom adsorbed
            bad_H += 1
    if bad_H != 0:
        print("{} filtered out: wrong H connectivity!".format(get_graph_formula(graph, ENCODER.categories_[0]), bad_H))
        return False
    else:
        return True

def C_connectivity_filter(graph: torch_geometric.data.Data):
    """
    Graph filter function that checks that each carbon atom in the molecule
    has at maximum 4 bonds.
    Args:
        graph(torch.geometric.data.Data): Graph object representation
    Returns:
        (bool): True=Correct connectivity for all C atoms
                False=Wrong connectivity for at least one C atom
    """
    # 1) Find index of C nodes
    C_nodes_indexes = []
    for i in range(graph.num_nodes):
        if graph.x[i, 2] == 1:  # 2 is the index of H in the one-hot encoder
            C_nodes_indexes.append(i)
    # 2) Apply correctness criterion to all H atoms
    # NB Just one bad connectivity makes the whole graph wrong
    bad_C = 0
    CHONS = {2, 5, 7, 9, 15} # indexes of C, H, O, N, S in the encoder
    for index in C_nodes_indexes:
        counter = 0  # edges between C and atoms in (C, H, O, N, S)
        for j in range(graph.num_edges):
            if index == graph.edge_index[0, j]:  # NB: Each edge repeated twice in order to have undirected
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0].item()  # most delicate part!
                if  other_atom in CHONS: 
                    counter += 1
        if counter > 4: # maximum 4 connections for C in a molecule
            bad_C += 1
    if bad_C != 0:
        print("{} filtered out: wrong C connectivity ({} edges)!".format(get_graph_formula(graph, ENCODER.categories_[0]), bad_C))
        return False
    else:
        return True

def global_filter(graph: torch_geometric.data.Data):
    """
    Filter function sum of single_fragment and H_connectivity.
    Args: 
        graph(pytorch_geometric.data.Data): graph object representation.
    Returns:
        (bool): True: both conditions are True
                False: other cases
    """
    condition1 = single_fragment_filter(graph)
    condition2 = H_connectivity_filter(graph)
    condition3 = C_connectivity_filter(graph)
    return (condition1 and condition2) and condition3

def is_chiral(graph: torch_geometric.data.Data):
    """Filter for chiral molecules (OR STEREOISOMERS?)

    Args:
        graph (_type_): _description_
    Returns:
        (bool)
    """
    # 1) Get molecule from adsorption configuration
    molecule = extract_adsorbate(graph)
    # 2) Find C atoms (potential stereocenters)
    C_index = ELEMENT_LIST.index("C")
    C_encoder = [0] * NODE_FEATURES
    C_encoder[C_index] = 1
    C_encoder = torch.tensor(C_encoder)
    C_nodes_list = []
    for node in range(molecule.num_nodes):
        index = torch.where(molecule.x[node, :] == 1)[0].item()
        if index == C_index:
            C_nodes_list.append(node)
    chiral_nodes = 0
    for C_atom in C_nodes_list:
        exploded_graph = explode_graph(molecule, C_atom)
        nx_graph = convert_gpytorch_to_networkx(exploded_graph)
        fragments = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
        num_of_fragments = len(fragments)
        diff_fragments = num_of_fragments
        iso_list = []
        if num_of_fragments == 4:
            #test with 4
            test_index = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
            for tuple in test_index:
                iso_list.append(nx.is_isomorphic(fragments[tuple[0]], fragments[tuple[1]]))
                

        #fragments = len(list(nx.connected_components(nx_graph)))
        
    return None

def explode_graph(graph: torch_geometric.data.Data, removed_node: int):
    """Explode graph into fragments keeping out the selected node

    Args:
        graph (torch_geometric.data.Data): _description_
        node_index (int): _description_

    Returns:
        _type_: _description_
    """
    # 1) Initialize graph feature matrix and connection for the new graph
    x = torch.zeros((graph.num_nodes-1, NODE_FEATURES))
    links = [graph.edge_index[1,i] for i in range(graph.num_edges) if graph.edge_index[0,i] == removed_node]
    edge = torch.zeros((2, graph.num_edges - 2 * len(links)))
    # 2) Find new indeces
    y = [None] * graph.num_nodes
    counter = 0
    for i in range(graph.num_nodes):
        if i != removed_node:
            y[i] = counter
            counter += 1
    def ff(node_index: int):
        return y[node_index]
    # 3) Remove connections between target node and other nodes
    edge_list = []
    edge_index = []
    for link in range(graph.num_edges):
        nodes = graph.edge_index[:, link]
        switch = 0
        for node in nodes:
            if node == removed_node:
                switch = 1
        if switch == 0:
            edge_list.append(nodes)
            edge_index.append(link)
        switch = 0
    # 4) Define new graph (delicate part)
    for node_index in range(x.shape[0]):
        if node_index < removed_node:
            xx = torch.where(graph.x[node_index, :] == 1)[0]
            x[node_index, xx] = 1
        elif node_index >= removed_node:
            xx = torch.where(graph.x[node_index+1, :] == 1)[0]
            x[node_index, xx] = 1
    for j in range(2):
        for k in range(edge.shape[1]):
            edge[j, k] = (ff(int(edge_list[k][j].item())))
    exploded_graph = torch_geometric.data.Data(x, edge)
    return exploded_graph
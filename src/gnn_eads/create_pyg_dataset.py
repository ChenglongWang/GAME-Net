import os
from torch_geometric.data import InMemoryDataset, Data
import torch

from ase.db import connect
import networkx as nx

from gnn_eads.graph_filters import isomorphism_test_2, adsorption_filter, H_connectivity_filter, C_connectivity_filter, single_fragment_filter
from gnn_eads.graph_tools import extract_adsorbate
from gnn_eads.functions import atoms_to_pyggraph


def convert_to_networkx(data: Data) -> nx.Graph:
    """
    Convert graph in PyG format to networkx format.
    Since edges are undirected, we only need to add one direction.

    Args:
        data (Data): PyG Data object.

    Returns:
        G (nx.Graph): Networkx graph.
    
    """
    edge_index = data.edge_index.cpu().numpy()
    edges = set()
    for i in range(edge_index.shape[1]):
        edge = tuple(sorted([edge_index[0, i], edge_index[1, i]]))
        edges.add(edge)
    nx_graph = nx.Graph()
    for edge in edges:
        nx_graph.add_edge(edge[0], edge[1])
    return nx_graph

def detect_ring_nodes(data: Data) -> set:
    """
    Return indices of the nodes in te PyG Data object that are part of a ring.

    Args:
        data (Data): PyG Data object.
    
    Returns:
        ring_nodes (set): Set of indices of the nodes that are part of a ring.    
    """
    nx_graph = convert_to_networkx(data)
    cycles = list(nx.cycle_basis(nx_graph))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    return ring_nodes


class AdsGraphDataset(InMemoryDataset):
    """
    Dataset class for the adsorption graphs.
    It generates the graphs from the ASE database and applies the graph filters.
    The graphs are stored in the PyG format.

    Args:
        root (str): Root directory where the dataset should be saved.
        dataset_name (str): Name of the PyG dataset that will be generated.
        ase_database_path (str): Path to the ASE database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation 
                             in the following format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "metal_hops": int},
                             "features": {"encoder": OneHotEncoder, "adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool}}
        target (str): Target property to be predicted. The label should be a column in the ASE database.
                      example: "e_ads", "gnn_target"
    """

    def __init__(self,
                 root: str,
                 dataset_name: str,
                 ase_database_path: str,
                 graph_params: dict,
                 target: str):
        self.root = root
        self.ase_database_path = ase_database_path
        self.graph_structure_params = graph_params["structure"]
        self.graph_features_params = graph_params["features"]    
        self.one_hot_encoder_elements = graph_params["features"]["encoder"]
        self.target = target
        self.database_size = 0 
        self.output_path = os.path.join(root, dataset_name)
        # Filter counters
        self.counter_isomorphism = 0
        self.counter_H_filter = 0
        self.counter_C_filter = 0
        self.counter_fragment_filter = 0
        self.counter_adsorption_filter = 0
        # Filter bins
        self.bin_isomorphism = []
        self.bin_H_filter = []
        self.bin_C_filter = []
        self.bin_fragment_filter = []
        self.bin_adsorption_filter = []
        # Node features
        self.node_feature_list = list(self.one_hot_encoder_elements.categories_[0])
        self.node_dim = len(graph_params["features"]["encoder"].categories_[0])
        if graph_params["features"]["adsorbate"]:
            self.node_dim += 1
            self.node_feature_list.append("Adsorbate")
        if graph_params["features"]["ring"]:
            self.node_dim += 1
            self.node_feature_list.append("Ring")
        if graph_params["features"]["aromatic"]:
            self.node_dim += 1
            self.node_feature_list.append("Aromatic")
        if graph_params["features"]["radical"]:
            self.node_dim += 1
            self.node_feature_list.append("Radical")
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])    
        print(self.processed_paths[0])


    @property
    def raw_file_names(self): 
        return self.ase_database_path
    
    @property
    def processed_file_names(self): 
        return self.output_path
    
    def download(self):
        pass
    
    def process(self):  
        molecule_elements = ["C", "H", "O", "N", "S"]
        elements_list = list(self.one_hot_encoder_elements.categories_[0]) 
        CHONS = [elements_list.index(element) for element in molecule_elements]
        data_list = []
        db = connect(self.ase_database_path)
        # Iterate over all rows in ASE database and create PyG Data objects
        for row in db.select('calc_type=adsorption'):
            self.database_size += 1
            atoms_obj = row.toatoms()
            calc_type = row.get("calc_type")
            formula = row.get("formula")
            family = row.get("family")
            target = float(row.get(self.target))
            # Get primitive PyG graph structure (nodes and edges, no label)
            graph = atoms_to_pyggraph(atoms_obj, 
                                      self.graph_structure_params["tolerance"], 
                                      self.graph_structure_params["scaling_factor"],
                                      self.graph_structure_params["metal_hops"], 
                                      self.one_hot_encoder_elements)
            # NODE FEATURIZATION
            if self.graph_features_params["adsorbate"]:
                x_adsorbate = torch.zeros((graph.x.shape[0], 1))  # 1=adsorbate, 0=metal
                for i, node in enumerate(graph.x):
                    index = torch.where(node == 1)[0][0].item()
                    if index in CHONS:
                        x_adsorbate[i, 0] = 1
                graph.x = torch.cat((graph.x, x_adsorbate), dim=1)
            if self.graph_features_params["ring"]:
                x_ring = torch.zeros((graph.x.shape[0], 1))  # 1=ring, 0=no ring
                mol_graph, index_list = extract_adsorbate(graph, self.one_hot_encoder_elements)
                mol_ring_nodes = detect_ring_nodes(mol_graph)
                for node_index in mol_ring_nodes:
                    x_ring[index_list.index(node_index), 0] = 1
                graph.x = torch.cat((graph.x, x_ring), dim=1)                
            if self.graph_features_params["aromatic"]:
                # TODO: implement aromaticity detection
                # x_aromatic = torch.zeros((graph.x.shape[0], 1))  # 1=aromatic, 0=no aromatic/metal
                # graph.x = torch.cat((graph.x, x_aromatic), dim=1)
                pass
            if self.graph_features_params["radical"]:
                # TODO: implement radical detection
                # x_radical = torch.zeros((graph.x.shape[0], 1))  # 1=radical, 0=no radical/ metal
                # graph.x = torch.cat((graph.x, x_radical), dim=1)
                pass

            # EDGE FEATURIZATION 
            # TODO: implement edge features

            # GRAPH LABELLING
            y = torch.tensor(target, dtype=torch.float)
            graph.target, graph.y = y, y
            graph.formula = formula
            graph.family = family
            graph.type = calc_type

            # FILTERING
            if adsorption_filter(graph, self.one_hot_encoder_elements):
                pass
            else:
                print("{} ({}) filtered out: No catalyst representation.".format(formula, family))
                self.bin_adsorption_filter.append(graph)
                self.counter_adsorption_filter += 1
                continue    
            if H_connectivity_filter(graph, self.one_hot_encoder_elements):
                pass
            else:
                print("{} ({}) filtered out: Wrong H connectivity within the adsorbate.".format(formula, family))
                self.bin_H_filter.append(graph)
                self.counter_H_filter += 1  
                continue
            if C_connectivity_filter(graph, self.one_hot_encoder_elements):
                pass
            else:
                print("{} ({}) filtered out: Wrong C connectivity within the adsorbate.".format(formula, family))
                self.bin_C_filter.append(graph)
                self.counter_C_filter += 1  
                continue
            if single_fragment_filter(graph, self.one_hot_encoder_elements):
                pass
            else:
                print("{} ({}) filtered out: Fragmented adsorbate.".format(formula, family))
                self.bin_fragment_filter.append(graph)
                self.counter_fragment_filter += 1  
                continue
            if isomorphism_test_2(graph, data_list, 0.01):  
                pass
            else:
                print("{} ({}) filtered out: Isomorphic to another graph.".format(formula, family))
                self.bin_isomorphism.append(graph)
                self.counter_isomorphism += 1
                continue
            data_list.append(graph)
            print("{} ({}) added to dataset".format(formula, family))
        print("Graph dataset size: {}".format(len(data_list)))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
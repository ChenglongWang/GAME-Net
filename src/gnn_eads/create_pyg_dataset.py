""" Module containing the class for the generation of the PyG dataset from the ASE database."""

import os

from torch_geometric.data import InMemoryDataset, Data
from torch import zeros, where, cat, load, save, tensor
import torch
from ase.db import connect
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from ase.atoms import Atoms
import networkx as nx

from gnn_eads.graph_filters import adsorption_filter, H_connectivity_filter, C_connectivity_filter, single_fragment_filter
from gnn_eads.graph_tools import extract_adsorbate
from gnn_eads.functions import atoms_to_pyggraph


def pyg_dataset_id(ase_database_path: str, 
                   graph_params: dict) -> str:
    """
    Provide dataset string identifier based on the provided graph parameters.
    
    Args:
        ase_database_path (str): Path to the ASE database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation 
                             in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "second_order_nn": int},
                             "features": {"encoder": OneHotEncoder, "adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool}}
    Returns:
        dataset_id (str): PyG dataset identifier.
    """
    # name of the ASE database (*.db file)
    id = ase_database_path.split("/")[-1].split(".")[0]
    # extract graph structure parameters
    structure_params = graph_params["structure"]
    tolerance = str(structure_params["tolerance"]).replace(".", "")
    scaling_factor = str(structure_params["scaling_factor"]).replace(".", "")
    metal_hops = str(structure_params["second_order_nn"])
    # extract node features parameters
    features_params = graph_params["features"]
    adsorbate = str(features_params["adsorbate"])
    ring = str(features_params["ring"])
    aromatic = str(features_params["aromatic"])
    radical = str(features_params["radical"])
    facet = str(features_params["facet"])
    target = graph_params["target"]
    # id convention: database name + target + all features. float values converted to strings and "." is removed
    dataset_id = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(id, target, tolerance, scaling_factor, metal_hops, adsorbate, ring, aromatic, radical, facet)
    return dataset_id


def get_radical_atoms(atoms_obj: Atoms, 
                      molecule_elements: list[str]) -> list[int]:
    """
    Detect atoms in the molecule which are radicals.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in molecule_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(conn_mol, Chem.SANITIZE_FINDRADICALS  ^ Chem.SANITIZE_SETHYBRIDIZATION)
    num_radical_electrons = [atom.GetNumRadicalElectrons() for atom in conn_mol.GetAtoms()]
    radical_atoms = [atom.GetIdx() for atom in conn_mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0]
    return radical_atoms


def detect_ring_nodes(data: Data) -> set:
    """
    Return indices of the nodes in the PyG Data object that are part of a ring.
    To do so, the graph is converted to a networkx graph and the cycle basis is computed.

    Args:
        data (Data): PyG Data object.
    
    Returns:
        ring_nodes (set): Set of indices of the nodes that are part of a ring.    
    """
    edge_index = data.edge_index.cpu().numpy()
    edges = set()
    for i in range(edge_index.shape[1]):
        edge = tuple(sorted([edge_index[0, i], edge_index[1, i]]))
        edges.add(edge)
    nx_graph = nx.Graph()
    for edge in edges:
        nx_graph.add_edge(edge[0], edge[1])
    cycles = list(nx.cycle_basis(nx_graph))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    return ring_nodes


def get_aromatic_atoms(atoms_obj: Atoms, 
                       molecule_elements: list[str]) -> list[int]:
    """
    Get indices of aromatic atoms in an ase atoms object with RDKit.

    Args:
        atoms_obj: ASE atoms object

    Returns:
        aromatic_atoms: list of aromatic atoms indices
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in molecule_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineBonds(conn_mol)
    aromatic_atoms = [atom.GetIdx() for atom in conn_mol.GetAtoms() if atom.GetIsAromatic()]
    return aromatic_atoms


def isomorphism_test(graph: Data, 
                     graph_list: list, 
                     eps: float=0.01) -> bool:
    """
    Perform isomorphism test for the input graph before including it in the final dataset.
    Test based on graph formula and energy difference.

    Args:
        graph (Data): Input graph.
        graph_list (list): graph list against which the input graph is tested.
        eps (float): tolerance value for the energy difference in eV. Default to 0.01 eV.
        grwph: data graph as input
    Returns:
        bool: Whether the graph passed the isomorphism test.
    """
    if len(graph_list) == 0:
        return True
    formula = graph.formula  # formula provided by ase 
    family = graph.family
    facet = graph.facet
    energy = graph.y
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    for rival_graph in graph_list:
        c1 = num_edges == rival_graph.num_edges
        c2 = num_nodes == rival_graph.num_nodes
        c3 = formula == rival_graph.formula
        c4 = np.abs(energy - rival_graph.y) < eps
        c5 = family == rival_graph.family
        c6 = facet == rival_graph.facet
        if c1 and c2 and c3 and c4 and c5 and c6:
            return False
        else:
            continue
    return True


class AdsorptionGraphDataset(InMemoryDataset):
    """
    Generate graph dataset representing molecules adsorbed on metal surfaces.
    It generates the graphs from the provided ase database and conversion settings.
    The graphs are stored in the torch_geometric.data.Data type.

    Args:
        ase_database_name (str): Path to the ase database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "metal_hops": int},
                             "features": {"adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool, "facet": bool}, 
                             "target": str}
        database_key (str): Key to access specific items of the ase database. Default to "calc_type=adsorption".
        
    Notes:
        - "target" in graph_params must be a category of the ase database.
        - The generated dataset is stored in the same directory of the ase database.
        - Each graph object has two labels: graph.y and graph.target. Originally they are equal, 
          but during the trainings graph.target represents the 
          original value (e.g., adsorption energy in eV), while graph.y is the scaled value (e.g.,
          unitless scaled adsorption energy).

    Example:
        Generate graph dataset containing only adsorption systems on Pt(111) surface, 
        with adsorbate, ring, aromatic, radical and facet features, and e_ads_dft as target.
        >>> graph_params = {"structure": {"tolerance": 0.5, "scaling_factor": 1.5, "metal_hops": False},
                            "features": {"adsorbate": True, "ring": True, "aromatic": True, "radical": True, "facet": True},
                            "target": "e_ads_dft"}
        >>> ase_database_path = "path/to/ase/database"
        >>> dataset = AdsorptionGraphDataset(ase_database_path, graph_params, "calc_type=adsorption,facet=fcc(111),metal=Pt")
    """

    def __init__(self,
                 ase_database_path: str,
                 graph_params: dict, 
                 database_key: str="calc_type=adsorption"):
        self.root = os.path.dirname(ase_database_path)
        self.ase_database_path = ase_database_path
        self.graph_structure_params = graph_params["structure"]
        self.graph_features_params = graph_params["features"]    
        self.target = graph_params["target"]
        self.database_key = database_key
        self.database_size = 0 
        self.dataset_id = pyg_dataset_id(self.ase_database_path, graph_params)
        self.output_path = os.path.join(self.root, self.dataset_id)
        # Construct one-hot encoder for chemical elements and surface orientation (based on the accessed data in the database)
        db = connect(self.ase_database_path)
        self.elements_list, self.surface_orientation_list = [], []
        for row in db.select(database_key):
            chemical_symbols = set(row.toatoms().get_chemical_symbols())    
            for element in chemical_symbols:
                if element not in self.elements_list:
                    self.elements_list.append(element)
            surface_orientation = row.get("facet")
            if surface_orientation not in self.surface_orientation_list:
                self.surface_orientation_list.append(surface_orientation)
        self.molecule_elements = [elem for elem in self.elements_list if elem in ["C", "H", "O", "N", "S"]]
        self.one_hot_encoder_elements = OneHotEncoder().fit(np.array(self.elements_list).reshape(-1, 1)) 
        self.one_hot_encoder_facets = OneHotEncoder().fit(np.array(self.surface_orientation_list).reshape(-1, 1))
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
        self.bin_unconverted_atoms_objects = []  # unsolved issue with ase
        # Node features
        self.node_feature_list = list(self.one_hot_encoder_elements.categories_[0])
        self.node_dim = len(self.node_feature_list)
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
        if graph_params["features"]["facet"]:
            self.node_dim += len(self.one_hot_encoder_facets.categories_[0])
            self.node_feature_list += list(self.one_hot_encoder_facets.categories_[0])
        super().__init__(root=os.path.dirname(ase_database_path))
        self.data, self.slices = load(self.processed_paths[0])    

    @property
    def raw_file_names(self): 
        return self.ase_database_path
    
    @property
    def processed_file_names(self): 
        return self.output_path
    
    def download(self):
        pass
    
    def process(self):  
        db = connect(self.ase_database_path)    
        elements_list = list(self.one_hot_encoder_elements.categories_[0]) 
        CHONS = [elements_list.index(element) for element in self.molecule_elements]
        data_list = []
        for row in db.select(self.database_key):
            self.database_size += 1
            atoms_obj = row.toatoms()
            calc_type = row.get("calc_type")
            formula = row.get("formula")
            family = row.get("family")
            metal = row.get("metal")
            facet = row.get("facet")
            target = float(row.get(self.target))
            print("Processing {} ({})".format(formula, family))
            # Get primitive PyG graph structure (nodes and edges, no label)
            try:
                graph = atoms_to_pyggraph(atoms_obj, 
                                      self.graph_structure_params["tolerance"], 
                                      self.graph_structure_params["scaling_factor"],
                                      self.graph_structure_params["second_order_nn"], 
                                      self.one_hot_encoder_elements, 
                                      self.molecule_elements)
            except:
                self.bin_unconverted_atoms_objects.append(atoms_obj)
                print("{} ({}) Error in graph structure generation.".format(formula, family))
                continue
            # NODE FEATURIZATION (definition of graph.x tensor)
            if self.graph_features_params["adsorbate"]:
                x_adsorbate = zeros((graph.x.shape[0], 1))  # 1=adsorbate, 0=metal
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    x_adsorbate[i, 0] = 1 if index in CHONS else 0
                graph.x = cat((graph.x, x_adsorbate), dim=1)
            if self.graph_features_params["ring"] and family:
                x_ring = zeros((graph.x.shape[0], 1))  # 1=ring, 0=no ring
                mol_graph, index_list = extract_adsorbate(graph, self.one_hot_encoder_elements)
                mol_ring_nodes = detect_ring_nodes(mol_graph)
                for node_index in mol_ring_nodes:
                    x_ring[index_list.index(node_index), 0] = 1
                graph.x = cat((graph.x, x_ring), dim=1)                
            if self.graph_features_params["aromatic"]:
                x_aromatic = torch.zeros((graph.x.shape[0], 1))  # 1=aromatic, 0=no aromatic/metal
                ring_descriptor_index = self.node_feature_list.index("Ring")  # double-check: aromatic atom -> ring atom
                if len(torch.where(graph.x[:, ring_descriptor_index] == 1)[0]) == 0:
                    graph.x = torch.cat((graph.x, x_aromatic), dim=1)
                else: # presence of rings
                    try: 
                        aromatic_atoms = get_aromatic_atoms(atoms_obj, ["C", "H", "N", "O", "S"])
                    except:
                        print("{} ({}) Error in aromatic detection.".format(formula, family))
                        continue
                    for index, node in enumerate(graph.x):
                        if node[ring_descriptor_index] == 0:  # atom not in a ring
                            x_aromatic[index, 0] = 0
                        else:  
                            if index in aromatic_atoms:
                                x_aromatic[index, 0] = 1 
                    graph.x = torch.cat((graph.x, x_aromatic), dim=1)
            if self.graph_features_params["radical"]:
                x_radical = torch.zeros((graph.x.shape[0], 1))  # 1=radical, 0=no radical/ metal
                radical_atoms = get_radical_atoms(atoms_obj, self.molecule_elements)
                print(radical_atoms)
                for index, node in enumerate(graph.x):
                    if index in radical_atoms:
                        x_radical[index, 0] = 1
                graph.x = torch.cat((graph.x, x_radical), dim=1)
            if self.graph_features_params["facet"]:
                x_facet = zeros((graph.x.shape[0], len(self.one_hot_encoder_facets.categories_[0])))
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    facet_index = list(self.one_hot_encoder_facets.categories_[0]).index(facet)
                    if index not in CHONS:
                        x_facet[i, facet_index] = 1
                graph.x = cat((graph.x, x_facet), dim=1)

            #node checking
            if graph.x.shape[1] != self.node_dim:
                raise ValueError("Node dimension mismatch: {} vs {}".format(graph.x.shape[1], self.node_dim))

            # EDGE FEATURIZATION 
            # TODO: implement edge features (and maybe hyperedges)

            # GRAPH LABELLING
            y = tensor(target, dtype=torch.float)
            graph.target, graph.y = y, y
            graph.formula = formula
            graph.family = family
            graph.type = calc_type
            graph.metal = metal
            graph.facet = facet

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
            if isomorphism_test(graph, data_list, 0.01):  
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
        save((data, slices), self.processed_paths[0])
        print("Dataset name: {}".format(self.processed_paths[0]))
        self.print_summary()
    
    
    def print_summary(self):
        database_name = "ASE database: {}\n".format(os.path.abspath(self.ase_database_path))
        selection_key = "Selection key: {}\n".format(self.database_key)
        database_size = "ASE database size: {}\n".format(self.database_size)
        graph_dataset_size = "Graph dataset size: {}\n".format(len(self.data))
        filtered_data = "Filtered data: {} ({:.2f}%)\n".format(self.database_size - len(self.data), 100 * (self.database_size - len(self.data)) / self.database_size)
        graph_dataset_path = "Graph dataset path: {}\n".format(os.path.abspath(self.output_path))
        print(database_name + selection_key + database_size + graph_dataset_size + filtered_data + graph_dataset_path)
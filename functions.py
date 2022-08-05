"""Functions for converting DFT data to graphs and for learning process purposes."""

__author__ = "Santiago Morandi"

from paths import paths
from itertools import product
import math
from collections import namedtuple

from sklearn.preprocessing import OneHotEncoder
from pyRDTP.geomio import file_to_mol
from pyRDTP.molecule import Molecule
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.io.vasp import Outcar
 
from constants import METALS, MOL_ELEM, FG_RAW_GROUPS, VORONOI_TOLERANCE, CORDERO,  ENCODER

def split_percentage(splits: int, test: bool=True) -> tuple[int]:
    """Return split percentage of the train, validation and test sets.

    Args:
        split (int): number of initial splits of the entire initial dataset

    Returns:
        a, b, c: train, validation, test percentage of the sets.
    """
    if test:
        a = int(100 - 200 / splits)
        b = math.ceil(100 / splits)
        return a, b, b
    else:
        return int((1 - 1/splits) * 100), math.ceil(100 / splits)

def mol_to_ensemble(molecule: Molecule,
                    voronoi_tolerance: float=VORONOI_TOLERANCE,
                    atom_rad_dict: dict=CORDERO, 
                    second_order: bool=False) -> Molecule:
    """
    Extract adsorbate + interacting metal atoms from the whole cell.
    Function based on pyRDTP.
    Args:
        molecule(pyRDTP.molecule.Molecule): molecule object.
        voronoi_tolerance(float): for connectivity search in pyRDTP.
        atom_rad_dict (dict): atomic radii dictionary (e.g., "ag": 0.8)
        second_order (bool): whether including metal 2nd order neighbours. Default to False
    Returns:
        new_molecule(pyRDTP.molecule.Molecule): adsorbate + interacting metal atoms ensemble.
    """
    # 1) Define whole connectivity in the cell
    molecule = connectivity_search_voronoi(molecule, tolerance=voronoi_tolerance, metal_rad_dict=atom_rad_dict)
    # 2) Create Molecule object with adsorbate and interacting metal atoms
    new_atoms = []
    non_metal_atoms = [atom for atom in molecule.atoms if atom.element not in METALS]
    # 3) Collect atoms
    for atom in non_metal_atoms:
        for neighbour in atom.connections + [atom]:
            if neighbour not in new_atoms:
                new_atoms.append(neighbour)
    # 3b) Collect metal neighbours of the metal atoms directly in contact with adsorbate
    if second_order: 
        for atom in new_atoms:
            if atom in METALS:
                for neighbour in atom.connections + [atom]:
                    if neighbour not in new_atoms:
                        new_atoms.append(neighbour)
            else:
                pass
    new_atoms = [atom.copy() for atom in new_atoms]
    new_molecule = Molecule("")
    new_molecule.atom_add_list(new_atoms)
    new_molecule.connection_clear()
    new_molecule.cell_p_add(molecule.cell_p.copy())
    # 4) Define connectivity of the final ensemble
    new_molecule = connectivity_search_voronoi(new_molecule, tolerance=voronoi_tolerance)
    return new_molecule

def mol_to_ensemble_CAVITY(molecule: Molecule,
                           voronoi_tolerance: float,
                           metal_rad_dict: dict) -> Molecule:
    """
    Extract adsorbate + interacting metal atoms from the whole system
    Function based on pyRDTP. FOR SAC CAVITIES
    Args:
        molecule(pyRDTP.molecule.Molecule): molecule object.
        tolerance(float): parameter for connectivity search 
                          algorithm in pyRDTP.
    Returns:
        new_molecule(pyRDTP.molecule.Molecule): ensemble of adsorbate + 
                                                interacting metal atoms
    """
    # 1) define whole connectivity in the cell
    molecule = connectivity_search_voronoi(molecule, tolerance=voronoi_tolerance, metal_rad_dict=metal_rad_dict)
    # 2) create Molecule object with adsorbate and interacting metal atoms
    new_atoms = []
    metal_atom = [atom for atom in molecule.atoms if atom.element in METALS]
    for neighbour in metal_atom[0].connections + [metal_atom[0]]:
        if neighbour not in new_atoms:
            new_atoms.append(neighbour)
        for second_neighbour in neighbour.connections + [neighbour]:
            if second_neighbour not in new_atoms:
                new_atoms.append(second_neighbour)
            # for third_neighbour in second_neighbour.connections + [second_neighbour]:
            #     if third_neighbour not in new_atoms:
            #         new_atoms.append(third_neighbour)
    new_atoms = [atom.copy() for atom in new_atoms]
    new_molecule = Molecule("")
    new_molecule.atom_add_list(new_atoms)
    new_molecule.connection_clear()
    new_molecule.cell_p_add(molecule.cell_p.copy())
    new_molecule = connectivity_search_voronoi(new_molecule, tolerance=voronoi_tolerance)    
    return new_molecule

def ensemble_to_graph(molecule: Molecule, 
                      second_order: bool=False) -> nx.Graph:
    """
    Convert pyRDTP Molecule to NetworkX graph.
    If second order neighbours are included, the metal-metal connnectons are kept.
    If only nearest metal atoms included, this action is not performed.
    Args:
        molecule(pyRDTP.molecule.Molecule): molecule object
    Returns:
        NetworkX graph object of the input molecule    
    """
    elem_lst = tuple([atom.element for atom in molecule.atoms])
    connections_1 = []
    connections_2 = []
    molecule.atom_index()
    for atom in molecule.atoms:
        for neighbour in atom.connections:
            if (atom.element == neighbour.element and atom.element in METALS) and second_order == False:
                continue  # Neglect metal-metal connections
            connections_1.append(atom.index)
            connections_2.append(neighbour.index)
    mol_graph = nx.Graph()
    for index, elem in enumerate(elem_lst):
        mol_graph.add_node(index, element=elem)
    for connection in zip(connections_1, connections_2):
        mol_graph.add_edge(*connection)
    return mol_graph, (elem_lst, (tuple(connections_1), tuple(connections_2)))

def get_energy(dataset: str) -> dict:
    """
    Extract the raw energy label for each sample of the dataset from the energies.dat file.    
    Args:
        dataset(str): Dataset's title.
    Returns:
        ener_dict(dict): Dictionary with raw total energy (sigma->0) [eV].    
    """
    with open(paths[dataset]['ener'], 'r') as infile:
        lines = infile.readlines()
    ener_dict = {}
    for line in lines:        
        split = line.split()
        ener_dict[split[0]] = float(split[1])
    return ener_dict

def get_structures(dataset: str) -> dict:
    """
    Extract the structure for each sample of the dataset from the 
    CONTCAR files in the "structures" folder.
    Args:
        dataset(str): Dataset's title.
    Returns:
        mol_dict(dict): Dictionary with pyRDTP.Molecule objects for each sample.  
    """
    mol_dict = {}
    for contcar in paths[dataset]['geom'].glob('./*.contcar'):
        mol_dict[contcar.stem] = file_to_mol(contcar, 'contcar', bulk=False)
    return mol_dict

def get_tuples(dataset: str,
               voronoi_tolerance: float,
               metal_radii_dict: dict, 
               second_order: bool=False) -> dict:
    """
    Generate a dictionary of namedtuple objects for each sample in the dataset.
    For datasets done by Santiago Morandi.
    Args:
        group_name(str): name of the dataset.
        voronoi_tolerance(float): parameter for the connectivity search in pyRDTP.
    Returns:
        ntuple_dict(dict): dictionary of namedtuple objects.    
    """
    if dataset not in FG_RAW_GROUPS:
        return "Dataset skipped: it doesn't belong to the FG-dataset."
    surf_ener = {key[:2]: value for key, value in get_energy("metal_surfaces").items()}
    mol_dict = get_structures(dataset)
    ener_dict = get_energy(dataset)
    ntuple = namedtuple(dataset, ['code', 'mol', 'graph', 'energy'])
    ntuple_dict = {}
    for key, mol in mol_dict.items():
        if dataset[0:3] != 'gas':
            splitted = key.split('-')
            elem, _, _ = splitted
            try:
                energy = ener_dict[key] - surf_ener[elem]
            except KeyError:
                print(f'{key} not found')
                continue
            try:
                mol = mol_to_ensemble(mol, voronoi_tolerance, metal_radii_dict, second_order)
                graph = ensemble_to_graph(mol, second_order)
            except ValueError:
                print(f'{key} not converting to graph')
                continue
        else: # Gas molecules
            energy = ener_dict[key]
            mol = mol_to_ensemble(mol, voronoi_tolerance, metal_radii_dict, second_order)
            graph = ensemble_to_graph(mol, second_order)
        ntuple_dict[key] = ntuple(code=key, mol=mol, graph=graph, energy=energy)
    return ntuple_dict

def export_tuples(filename: str,
                  tuple_dict: dict):
    """
    Write text file for GNN raw data from tuple dictionary representing the 
    dataset.
    Args:
        filename (str): file to write
        tuple_dict (tuple): tuple dictionary
    """
    with open(filename, 'w') as outfile:
        for code, inter in tuple_dict.items():
            lst_trans = lambda x: " ".join([str(y) for y in x])
            elems = " ".join([str(element) for element in inter.graph[1][0]])
            outfile.write(f'{code}\n')
            outfile.write(f'{lst_trans(inter.graph[1][0])}\n')
            outfile.write(f'{lst_trans(inter.graph[1][1][0])}\n')
            outfile.write(f'{lst_trans(inter.graph[1][1][1])}\n')
            outfile.write(f'{inter.energy}\n')

def geometry_to_graph_analysis(dataset:str):
    """
    Check that all adsorption samples in the dataset are correctly 
    converted to a graph.
    Args: 
        dataset(str): Dataset's title.
    Returns:  
        wrong_graphs(int): number of uncorrectly-converted samples, i.e., no metal atom is 
                           present as node in the graph representation.
        wrong_samples(list): list of the badly represented data.
        dataset_size(int): dataset size.
    """
    if dataset[:3] == "gas":
        print("{} dataset contains gas phase molecules".format(dataset))
        print("------------------------------------------")
        return "This function is just for datasets containing adsorption systems!"
    with open("./Data/{}/Dataset.dat".format(dataset)) as f:
        all_lines = f.readlines()
    dataset_size = int(len(all_lines)/5)
    lines = []
    labels = []
    for i in range(dataset_size):
        lines.append(all_lines[1 + 5*i])  # Read the second line of each graph (ex. "C H C H Ag")
        labels.append(all_lines[5*i])     # Read label of each sample (ex. "ag-4a01-a")
    for i in range(dataset_size):
        lines[i] = lines[i].strip("\n")
        lines[i] = lines[i].split()
        labels[i] = labels[i].strip("\n")
    new_list = [[]] * dataset_size
    wrong_samples = []
    for i in range(dataset_size):
        new_list[i] = [lines[i][j] for j in range(len(lines[i])) if lines[i][j] not in MOL_ELEM]
        if new_list[i] == []:
            wrong_samples.append(labels[i])
    wrong_graphs = int(new_list.count([]))
    print("Dataset Name: {}".format(dataset))
    print("Dataset Size: {}".format(dataset_size))
    print("Number of badly represented data: {}".format(wrong_graphs))
    print("Percentage of bad representations: {:.2f}%".format((wrong_graphs/dataset_size)*100))
    print("-------------------------------------------")
    return wrong_graphs, wrong_samples, dataset_size

def get_graph_formula(graph: Data,
                      categories,
                      metal_list: list=METALS) -> str:
    """ 
    Create a string label for the selected graph.
    String format: xxxxxxxxxxxxxx (len=14)
    CxHyOzNwSt-mmx
    Args:
        graph(torch_geometric.data.Data): graph object
        categories(list): list with element string labels
        metal_list(list): list of metal atoms string.
    Returns:
        label(str): label of the respective graph
    """
    element_list = []
    for i in range(graph.num_nodes):
        for j in range(graph.num_features):
            if graph.x[i, j] == 1:
                element_list.append(j)
    element_array = [0] * len(categories)
    for element in range(len(categories)):
        for index in element_list:
            if element == index:
                element_array[element] += 1
    element_array = list(element_array)
    element_array = [int(i) for i in element_array]
    element_dict = dict(zip(categories, element_array))
    label = ""
    ss = ""
    for key in element_dict.keys():
        if element_dict[key] == 0:
            pass
        else:
            label += key
            label += str(element_dict[key])
    for metal in metal_list:
        if metal in label:
            index = label.index(metal)
            ss = label[index:index+3]
    label = label.replace(ss, "")
    label += "-" + ss
    #label = label.replace("1", "")
    counter = 0
    for metal in metal_list:
        if metal in label:
            counter += 1
    if counter == 0:
        label += "(g)"
    # Standardize string length to 14
    diff = 14 - len(label)
    if diff > 0:
        extra_space = " " * diff
        label += extra_space
    return label

def get_number_atoms_from_label(formula:str,
                                H_count:bool=True) -> int:
    """Get the total number of atoms in the adsorbate from a graph formula
    got from get_graph_formula

    Args:
        formula (str): string representing the graph chemical formula
    """
    # 1) Remove everything after "-"
    n = 0
    my_list = ["0"]
    clr_form = formula.split('-')[0]
    for char in clr_form:
        if char.isalpha():
            test = 0
            n += int("".join(my_list))
            my_list = []
            if char == 'H':
                my_list.append("0")
                test = 1
            continue
        if test:
            continue
        my_list.append(char)
    n += int("".join(my_list))
    return n

def create_loaders(datasets:tuple,
                   split: int=5,
                   batch_size:int =32,
                   test:bool=True) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        datasets(tuple): tuple containing the HetGraphDataset objects.
        split(int): number of splits to generate train/val/test sets.
        batch_size(int): batch size. Default to 32.
        test(bool): Whether to generate train/val/test loaders or just train/val.    
    Returns:
        (tuple): tuple with dataloaders for training, validation and test.
    """
    train_loader = []
    val_loader = []
    test_loader = []
    for dataset in datasets:
        n_items = len(dataset)
        sep = n_items // split
        dataset.shuffle()
        if test == True:
            test_loader += (dataset[:sep])
            val_loader += (dataset[sep:sep*2])
            train_loader += (dataset[sep*2:])
        else:
            val_loader += (dataset[:sep])
            train_loader += (dataset[sep:])
    train_n = len(train_loader)
    val_n = len(val_loader)
    test_n = len(test_loader)
    total_n = train_n + val_n + test_n
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
    if test == True:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
        return (train_loader, val_loader, test_loader)
    else:
        print("Data split (train/val): {}/{} %".format(int(100*(split-1)/split), int(100/split)))
        print("Training data = {} Validation data = {} (Total = {})".format(train_n, val_n, total_n))
        return (train_loader, val_loader, None)

def scale_target(train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader=None,
                 mode: str='std',
                 verbose: bool=True,
                 test: bool=True):
    """
    Apply target scaling to the whole dataset using labels of
    training and validation sets.
    Args:
        train(torch_geometric.loader.DataLoader): training dataloader 
        val(torch_geometric.loader.DataLoader): validation dataloader
        test(torch_geometric.loader.DataLoader): test dataloader
    Returns:
        train, val, test: dataloaders with scaled target values
        mean_tv, std_tv: mean and std (standardization)
        min_tv, max_tv: min and max (normalization)
    """
    # 1) Get mean-std/min-max from train and validation sets
    y_list = []
    for graph in train_loader.dataset:
        y_list.append(graph.y.item())
    for graph in val_loader.dataset:
        y_list.append(graph.y.item())
    y_tensor = torch.tensor(y_list)
    mean_tv = y_tensor.mean(dim=0, keepdim=True)  # _tv stands for "train+validation sets"
    std_tv = y_tensor.std(dim=0, keepdim=True)
    max_tv = y_tensor.max()
    min_tv = y_tensor.min()
    # 2) Apply Scaling (Standardization or Normalization)
    for graph in train_loader.dataset:
        if mode == "std":
            graph.y = (graph.y - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.y - min_tv) / (max_tv - min_tv)
        else:
            pass
    for graph in val_loader.dataset:
        if mode == "std":
            graph.y = (graph.y - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.y - min_tv) / (max_tv - min_tv)
        else:
            pass
    if test == True:
        for graph in test_loader.dataset:
            if mode == "std":
                graph.y = (graph.y - mean_tv) / std_tv
            elif mode == "norm":
                graph.y = (graph.y - min_tv) / (max_tv - min_tv)
            else:
                pass
    if mode == "std":
        if verbose == True:
            print("Target Scaling (Standardization) applied successfully")
            print("(Train+Val) mean: {:.2f} eV".format(mean_tv.item()))
            print("(Train+Val) standard deviation: {:.2f} eV".format(std_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, mean_tv.item(), std_tv.item()
        else:
            return train_loader, val_loader, None, mean_tv.item(), std_tv.item()
    elif mode == "norm": 
        if verbose == True:
            print("Target Scaling (Normalization) applied successfully")
            print("(Train+Val) min: {:.2f} eV".format(min_tv.item()))
            print("(Train+Val) max: {:.2f} eV".format(max_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, min_tv.item(), max_tv.item()
        else:
            return train_loader, val_loader, None, min_tv.item(), max_tv.item()
    else:
        print("Target Scaling NOT applied")
        return train_loader, val_loader, test_loader, 0, 1

def connectivity_search_voronoi(molecule: Molecule,
                                tolerance:float=VORONOI_TOLERANCE,
                                metal_rad_dict:dict=CORDERO,
                                center:bool=False) -> Molecule:
    """
    Perform a connectivity search using the Voronoi tessellation. The method
    considers periodic boundary conditions and CORDERO Radius.
    Args:
        molecule(pyRDTP.molecule.Molecule): input molecule for which connectivity
            search is performed.
        tolerance (float, optional): Tolerance that will be added to the
            medium distance between two atoms.
        center (bool, optional): If True, the direct coordinates array of
            the molecule will be centered before the bond calculation to
            avoid errors in far from the box molecules. The coordinates
            of the molecule will not change.
    
    Notes:
        self.pairs will be fullfilled after the use of this method.
    Cite:
        Isayev, O. et al. Universal fragment descriptors for predicting
            properties of inorganic crystals. Nat. Commun. 8, 15679
            doi: 10.1038/ncomms15679 (2017).
    """
    if len(molecule.atoms) == 1:
        return molecule
    if center:
        cartesian_old = np.copy(molecule.coords_array('cartesian'))
        direct_old = np.copy(molecule.coords_array('direct'))
        molecule.move_to_box_center()
    coords_arr = np.copy(molecule.coords_array('direct'))
    coords_arr = np.expand_dims(coords_arr, axis=0)
    coords_arr = np.repeat(coords_arr, 27, axis=0)
    mirrors = [-1, 0, 1]
    mirrors = np.asarray(list(product(mirrors, repeat=3)))
    mirrors = np.expand_dims(mirrors, 1)
    mirrors = np.repeat(mirrors, coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors,
                                  (coords_arr.shape[0] * coords_arr.shape[1],
                                   coords_arr.shape[2]))
    corrected_coords = np.dot(corrected_coords, molecule.cell_p.direct)
    translator = np.tile(np.arange(coords_arr.shape[1]),
                         coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    true_arr = pairs_corr[:, 0] == pairs_corr[:, 1]
    true_arr = np.argwhere(true_arr)
    pairs_corr = np.delete(pairs_corr, true_arr, axis=0)
    dst_d = {}
    pairs_lst = []
    for pair in pairs_corr:
        elements = [molecule.atoms[index].element for index in pair]
        fr_elements = frozenset(elements)
        if fr_elements not in dst_d:
            dst_d[fr_elements] = metal_rad_dict[elements[0]]
            dst_d[fr_elements] += metal_rad_dict[elements[1]]
            dst_d[fr_elements] += tolerance
        if dst_d[fr_elements] >= molecule.distance(*pair, system='cartesian',
                                               minimum=True):
            pairs_lst.append(pair)
            molecule.atoms[pair[0]].connection_add(molecule.atoms[pair[1]])
    molecule.pairs = np.asarray(pairs_lst)
    if center:
        molecule.coords_update(cartesian_old, 'cartesian')
        molecule.coords_update(direct_old, 'direct')
    return molecule

def train_loop(model,
               device:str,
               train_loader: DataLoader,
               optimizer,
               loss_fn):
    """
    Helper function for model training over an epoch. 
    For each batch in the epoch, the following actions are performed:
    1) Move the batch to the selected device for training
    2) Forward pass through the GNN model and loss function computation
    3) Compute gradient of loss function wrt model parameters
    4) Update model parameters
    Args:
        model(): GNN model object.
        device(str): device on which training is performed.
        train_loader(): Training dataloader.
        optimizer(): optimizer used during training.
        loss_fn(): Loss function used for the training.
    Returns:
        loss_all(float): Mean Squared Error (MSE) of the whole epoch.   
    """
    model.train()  # Sets model in training mode
    loss_all = 0
    mae_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()                     # Set gradients of all tensors to zero
        loss = loss_fn(model(batch), batch.y)
        mae = F.l1_loss(model(batch), batch.y)    # For comparison with val/test data
        loss.backward()                           # Compute gradient of loss function wrt parameters
        loss_all += loss.item() * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs
        optimizer.step()                          # Update model parameters
    loss_all /= len(train_loader.dataset)
    mae_all /= len(train_loader.dataset)
    return loss_all, mae_all 

def test_loop(model,
              loader: DataLoader,
              device: str,
              std: float,
              mean: float=None, 
              scaled_graph_label: bool= True, 
              verbose: int=0) -> float:
    """
    Helper function for validation/testing.
    For each batch in the validation/test epoch, the following actions are performed:
    1) Set the GNN model in evaluation mode
    2) Move the batch to the selected device where the model is stored
    3) Compute the Mean Absolute Error (MAE)
    Args:
        model (): GNN model object.
        loader (Dataloader object): Dataset for validation/testing.
        device (str): device on which training is performed.
        std (float): standard deviation of the training+validation datasets [eV]
        mean (float): mean of the training+validation datasets [eV]
        scaled_graph_label (bool): whether the graph labels are in eV or in a scaled format.
        verbose (int): 0=no printing info 1=printing information
    Returns:
        error(float): Mean Absolute Error (MAE) of the test loader.
    """
    model.eval()   # Sets model in inference mode
    error = 0
    for batch in loader:
        batch = batch.to(device)
        if scaled_graph_label == False:  # label in eV
            error += (model(batch) * std + mean - batch.y).abs().sum().item()
        else:  #  Scaled label
            error += (model(batch) * std - batch.y * std).abs().sum().item()  
    error /= len(loader.dataset)  # Mean Absolute Error
    
    if verbose == 1:
        print("Dataset size = {}".format(len(loader.dataset)))
        print("Mean Absolute Error = {} eV".format(error))
    return error 

def get_mean_std_from_model(model_path:str) -> tuple[float]:
    """Get mean and standard deviation used for scaling the target values 
       from the selected trained model.

    Args:
        model_name (str): GNN model name
    
    Returns:
        mean, std (tuple[float]): mean and standard deviation for scaling the targets.
    """
    file = open("{}performance.txt".format(model_path), "r")
    lines = file.readlines()
    for line in lines:
        if "(train+val) mean" in line:
            mean = float(line.split()[-2])
        if "(train+val) standard deviation" in line:
            std = float(line.split()[-2])
    return mean, std

def contcar_to_graph(contcar_file: str,
                     voronoi_tolerance: float=VORONOI_TOLERANCE,
                     atomic_radius_dict: dict=CORDERO, 
                     one_hot_encoder=ENCODER, 
                     second_order: bool=False) -> Data:
    """Create graph representation from VASP CONTCAR file

    Args:
        contcar_file (str): Path to CONTCAR file.
        voronoi_tolerance (float, optional): Tolerance applied during the graph conversion.
        atomic_radius_dict (dict, optional): Atomic radius dictionary. Defaults to CORDERO.
        one_hot_encoder (_type_, optional): One-hot encoder. Defaults to encoder.

    Returns:
        graph (torch_geometric.data.Data): Graph object representing the system under study.
    """
    mol = file_to_mol(contcar_file, 'contcar', bulk=False)
    mol = mol_to_ensemble(mol, voronoi_tolerance, atomic_radius_dict, second_order=second_order)
    nx_graph = ensemble_to_graph(mol, second_order=second_order)
    elem = list(nx_graph[1][0])
    source = list(nx_graph[1][1][0])
    target = list(nx_graph[1][1][1])
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index)
    return graph

def contcar_to_graph_CAVITY(contcar_file: str,
                            voronoi_tolerance: float=VORONOI_TOLERANCE,
                            atomic_radius_dict: dict=CORDERO, 
                            one_hot_encoder=ENCODER) -> Data:
    """Create graph representation from VASP CONTCAR file

    Args:
        contcar_file (str): Path to CONTCAR file.
        voronoi_tolerance (float, optional): Tolerance applied during the graph conversion.
        atomic_radius_dict (dict, optional): Atomic radius dictionary. Defaults to CORDERO.
        one_hot_encoder (_type_, optional): One-hot encoder. Defaults to encoder.

    Returns:
        graph (torch_geometric.data.Data): Graph object representing the system under study.
    """
    mol = file_to_mol(contcar_file, 'contcar', bulk=False)
    mol = mol_to_ensemble_CAVITY(mol, voronoi_tolerance, atomic_radius_dict)
    nx_graph = ensemble_to_graph(mol)
    elem = list(nx_graph[1][0])
    source = list(nx_graph[1][1][0])
    target = list(nx_graph[1][1][1])
    elem_array = np.array(elem).reshape(-1, 1)
    elem_enc = one_hot_encoder.transform(elem_array).toarray()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    x = torch.tensor(elem_enc, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index)
    return graph

def get_graph_sample(system: str, 
                     surface: str= None,
                     voronoi_tolerance: float=VORONOI_TOLERANCE, 
                     atomic_radius_dict: dict=CORDERO, 
                     encoder: OneHotEncoder=ENCODER,
                     gas_mol: bool=False,
                     family: str=None, 
                     surf_multiplier: int=None, 
                     second_order: bool=False) -> Data:
    """ 
    Generate labelled graph samples from VASP files.
    
    Args: 
        system (str): path to the VASP folder containing the adsorption system
        surface (str): path to the VASP folder containing the empty surface
        voronoi_tolerance (float): tolerance applied during the conversion to graph
        atomic_radius_dict (dict): dictionary containing the atomic radii of elements
                                   for the conversion to graph
        encoder (OneHotEncoder): one-hot encoder used to represent atomic elements    
    Returns: 
        graph (Data): graph sample with label for comparison GNN-VASP.
    """
    system_energy = Outcar("{}/OUTCAR".format(system)).final_energy
    graph = contcar_to_graph("{}/CONTCAR".format(system),
                             voronoi_tolerance=voronoi_tolerance, 
                             atomic_radius_dict=atomic_radius_dict, 
                             one_hot_encoder=encoder, 
                             second_order=second_order) 
    graph.y = system_energy
    if gas_mol == False:
        surf_energy = Outcar("{}/OUTCAR".format(surface)).final_energy
        if surf_multiplier is not None:
            surf_energy *= surf_multiplier
        graph.y -= surf_energy  
    graph.formula = get_graph_formula(graph, ENCODER.categories_[0])
    if family is not None:
        graph.family = family
    return graph

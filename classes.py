"""Module containing classes implemented for the GNN definition"""

from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from constants import ENCODER, FAMILY_DICT, METALS
from graph_filters import global_filter, isomorphism_test
from graph_tools import plotter
from functions import get_graph_formula

class HetGraphDataset(InMemoryDataset):
    """
    InMemoryDataset is the abstract class for creating custom datasets.
    Each dataset gets passed a root folder which indicates where the dataset should
    be stored. The root folder is split up into 2 folders, a raw_dir where the dataset gets downloaded to,
    and the processed_dir, where the processed dataset is being saved.
    In order to create a InMemoryDataset class, four fundamental methods must be provided:
    - raw_file_names(): a list of files in the raw_dir which needs to be found in order to skip the download
    - file_names(): a list of files in the processed_dir which needs to be found in order to skip the processing
    - download(): download raw data into raw_dir
    - process(): process raw_data and saves it into the processed_dir
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(HetGraphDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # Raw dataset directory (plain text file from which graphs are generated)
        return self.root / 'Dataset.dat'  
    @property
    def processed_file_names(self):  # Processed dataset directory 
        return self.root / 'Dataset_tol025_sf15_2nd_order_iso2.dat' 

    def download(self):
        pass

    def process(self):  
        """
        If Dataset_pp.dat is not found in the directory, the
        class launches this method to process the raw data.
        NB: This method is run everytime the path given in processed_file_names
        is not present. Thus, if you want to regenerate the samples, you have to change the name 
        provided there (likely there is a more efficient way to regenerate the dataset).
        """
        data_list = []
        with open(self.raw_file_names, 'r') as infile:
            lines = infile.readlines()
        split_n = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)]
        splitted = split_n(lines, 5)  # Each sample is described by 5 text lines!
        for block in splitted:        # Each block represents a sample graph
            to_int = lambda x: [float(i) for i in x]
            _, elem, source, target, energy = block
            element_list = elem.split()
            # filter for removing graphs with no metal
            dataset_name = str(self.root).split("/")[-1]
            if dataset_name[:3] != "gas":
                counter = 0
                for element in element_list:
                    if element in METALS:
                        counter += 1
                if counter == 0:
                    continue                     
            #-----------------------------------------------
            elem_array = np.array(elem.split()).reshape(-1, 1)
            elem_enc = ENCODER.transform(elem_array).toarray()
            x = torch.tensor(elem_enc, dtype=torch.float)         # Node feature matrix
            edge_index = torch.tensor([to_int(source.split()),    # Edge list COO format
                                       to_int(target.split())],
                                       dtype=torch.long)       
            y = torch.tensor([float(energy)], dtype=torch.float)  # Graph label (Edft - Eslab)
            family = FAMILY_DICT[dataset_name]                    # Chemical family of the adsorbate/molecule
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family)
            graph_formula = get_graph_formula(data, ENCODER.categories_[0])
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family, formula=graph_formula)
            if global_filter(data):
                if isomorphism_test(data, data_list):
                     data_list.append(data)  
                #data_list.append(data)
            else:
                plotter(data)            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def dat_to_graph(graph_list:list):
    """Generate the respective graph object from text

    Args:
        graph_list (list): list of lines
    """
    
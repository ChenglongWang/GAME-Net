from paths import paths
from torch import load
from torch_geometric.loader import DataLoader

from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
import toml

from constants import ENCODER, FAMILY_DICT, METALS
from graph_filters import global_filter, isomorphism_test
from graph_tools import plotter
from functions import get_graph_formula
from paths import pre_id, post_id
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
        super(HetGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])       

    @property
    def raw_file_names(self):  # Raw dataset directory (plain text file from which graphs are generated)
        return self.root / pre_id 
    @property
    def processed_file_names(self):  # Processed dataset directory 
        return self.root / post_id 

    def download(self):
        pass

    def process(self):  
        """
        If the path provided in processed_file_names() is not existing, the
        class automatically runs this method to process the raw data.
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
            if global_filter(data):  # To ensure correct adsorbate representation in the graph
                if isomorphism_test(data, data_list):  # To ensure absence of duplicates graphs
                     data_list.append(data)  
            else:
                plotter(data)            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

aromatics_dataset = HetGraphDataset(paths['aromatics']['root'])
group2_dataset = HetGraphDataset(paths['group2']['root'])
group2b_dataset = HetGraphDataset(paths['group2b']['root'])
aromatics2_dataset = HetGraphDataset(paths['aromatics2']['root'])
carbamate_esters_dataset = HetGraphDataset(paths['carbamate_esters']['root'])
group3N_dataset = HetGraphDataset(paths['group3N']['root'])
group3S_dataset = HetGraphDataset(paths['group3S']['root'])
group4_dataset = HetGraphDataset(paths['group4']['root'])
amides_dataset = HetGraphDataset(paths['amides']['root'])
amidines_dataset = HetGraphDataset(paths['amidines']['root'])
oximes_dataset = HetGraphDataset(paths['oximes']['root'])
gas_amides_dataset = HetGraphDataset(paths['gas_amides']['root'])
gas_amidines_dataset = HetGraphDataset(paths['gas_amidines']['root'])
gas_aromatics_dataset = HetGraphDataset(paths['gas_aromatics']['root'])
gas_aromatics2_dataset = HetGraphDataset(paths['gas_aromatics2']['root'])
gas_group2_dataset = HetGraphDataset(paths['gas_group2']['root'])
gas_group2b_dataset = HetGraphDataset(paths['gas_group2b']['root'])
gas_group3N_dataset = HetGraphDataset(paths['gas_group3N']['root'])
gas_group3S_dataset = HetGraphDataset(paths['gas_group3S']['root'])
gas_carbamate_esters_dataset = HetGraphDataset(paths['gas_carbamate_esters']['root'])
gas_oximes_dataset = HetGraphDataset(paths['gas_oximes']['root'])
gas_group4_dataset = HetGraphDataset(paths['gas_group4']['root'])

FG_dataset = (group2_dataset,
               group2b_dataset,
               aromatics_dataset,
               aromatics2_dataset,
               amides_dataset,
               amidines_dataset,
               oximes_dataset,
               carbamate_esters_dataset,
               group3S_dataset,
               group3N_dataset,
               group4_dataset,
               gas_amides_dataset,
               gas_amidines_dataset,
               gas_aromatics_dataset,
               gas_aromatics2_dataset,
               gas_carbamate_esters_dataset,
               gas_group2_dataset,
               gas_group2b_dataset,
               gas_group3N_dataset,
               gas_group3S_dataset,
               gas_group4_dataset,
               gas_oximes_dataset) 

BM_dataset = load("./BM_dataset/Graph_dataset.pt")
BM_dataloader = DataLoader(BM_dataset)
from classes import HetGraphDataset
from paths import paths
from torch import load
from torch_geometric.loader import DataLoader

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
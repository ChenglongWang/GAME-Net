"""Script for the direct comparison of the GNN performance compared to DFT.
It works wirth systems containing the following elements: 
Adsorbate: C, H, O, N, S
Catalyst slab: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn
"""

import argparse
import sys
sys.path.insert(0, "../src")

import torch, toml
from gnn_eads.functions import structure_to_graph
from gnn_eads.functions import create_loaders, scale_target, train_loop, test_loop, EarlyStopper
from gnn_eads.graph_tools import extract_adsorbate
from gnn_eads.nets import PreTrainedModel
from gnn_eads.create_pyg_dataset import AdsorptionGraphDataset


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT system to graph and compare the GNN prediction to the DFT outcome.")
    PARSER.add_argument("-i", "--input", type=str, dest="input",
                        help="Path to the POSCAR file of the specific adsorption system/gas-molecule.")
    ARGS = PARSER.parse_args()

    # 1) Load pre-trained GNN model on CPU
    MODEL_PATH = "/homes/clwang/Code/gnn_eads/results/best_model_new_222510"
    model = PreTrainedModel(MODEL_PATH)

    hyperparameters = toml.load(ARGS.input)
    ase_database_path = hyperparameters["data"]["ase_database_path"]
    ase_database_key = hyperparameters["data"]["ase_database_key"]
    graph_settings = hyperparameters["graph"]
    train = hyperparameters["train"]
    architecture = hyperparameters["architecture"]
    # Select device
    device_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name: {} (GPU)".format(torch.cuda.get_device_name(0)))
        device_dict["name"] = torch.cuda.get_device_name(0)
        device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
        device_dict["CUDNN_version"] = torch.backends.cudnn.version()
        device_dict["CUDA_version"] = torch.version.cuda
    else:
        print("Device name: CPU")
        device_dict["name"] = "CPU"
    # Load graph dataset
    dataset = AdsorptionGraphDataset(ase_database_path, graph_settings, ase_database_key)
    train_loader, val_loader, test_loader = create_loaders(
        dataset, batch_size=1, split=train["splits"], test=True
    )

    # 3) Get GNN prediction
    model.model.to(device)
    error = 0
    for batch in test_loader:
        batch = batch.to(device)
        predict = model.evaluate(batch[0])
        print("Ads dft Energy: {} eV".format(predict))

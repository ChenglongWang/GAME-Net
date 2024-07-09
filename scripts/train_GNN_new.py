"""Perform GNN model training."""

import argparse
from pathlib import Path
from os.path import isdir
import time
import sys

sys.path.insert(0, "../src")

import torch

torch.backends.cudnn.deterministic = True
import toml
from torch_geometric.seed import seed_everything

seed_everything(42)

from gnn_eads.functions import create_loaders, scale_target, train_loop, test_loop, EarlyStopper
from gnn_eads.nets import FlexibleNet
from gnn_eads.post_training import create_model_report, plot_loss
from gnn_eads.constants import loss_dict, pool_seq_dict, conv_layer, sigma_dict, pool_dict
from gnn_eads.create_pyg_dataset import AdsorptionGraphDataset


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Perform a training process with the provided hyperparameter settings."
    )
    PARSER.add_argument(
        "-i", "--input", type=str, dest="i", help="Input toml file with hyperparameters for the learning process."
    )
    PARSER.add_argument("-o", "--output", type=str, dest="o", help="Output directory for the results.")
    ARGS = PARSER.parse_args()

    output_name = ARGS.i.split("/")[-1].split(".")[0] + f"_{time.strftime('%H%M%S')}"
    output_directory = Path(__file__).resolve().parent.parent/"results"

    # Upload training hyperparameters from toml file
    hyperparameters = toml.load(ARGS.i)
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
    print("dataset len:", len(dataset), type(dataset))
    dataset = dataset.shuffle()
    # Create train/validation/test dataloaders
    train_loader, val_loader, test_loader = create_loaders(
        dataset, batch_size=train["batch_size"], split=train["splits"], test=train["test_set"]
    )
    # Apply target scaling
    train_loader, val_loader, test_loader, mean, std = scale_target(
        train_loader, val_loader, test_loader, mode=train["target_scaling"], test=train["test_set"]
    )
    # Instantiate the GNN model
    model = FlexibleNet(
        dim=architecture["dim"],
        N_linear=architecture["n_linear"],
        N_conv=architecture["n_conv"],
        adj_conv=architecture["adj_conv"],
        in_features=dataset.node_dim,
        sigma=sigma_dict[architecture["sigma"]],
        bias=architecture["bias"],
        conv=conv_layer[architecture["conv_layer"]],
        pool=pool_dict[architecture["pool_layer"]],
        pool_ratio=architecture["pool_ratio"],
        pool_heads=architecture["pool_heads"],
    ).to(device)

    # Load optimizer, lr-scheduler, and early stopper
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train["lr0"],
        eps=train["eps"],
        weight_decay=train["weight_decay"],
        amsgrad=train["amsgrad"],
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train["factor"], patience=train["patience"], min_lr=train["minlr"]
    )
    if train["early_stopping"]:
        early_stopper = EarlyStopper(patience=train["es_patience"], start_epoch=train["es_start_epoch"])
    # Run iterative learning
    loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []
    t0 = time.time()
    for epoch in range(1, train["epochs"] + 1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]["lr"]
        loss, train_MAE = train_loop(model, device, train_loader, optimizer, loss_dict[train["loss_function"]])
        val_MAE = test_loop(model, val_loader, device, std)
        lr_scheduler.step(val_MAE)
        if train["test_set"]:
            test_MAE = test_loop(model, test_loader, device, std, mean)
            test_list.append(test_MAE)
            print(
                "Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV "
                "Test MAE: {:.4f} eV".format(epoch, lr, train_MAE * std, val_MAE, test_MAE)
            )
        else:
            print(
                "Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV ".format(
                    epoch, lr, train_MAE * std, val_MAE
                )
            )
        loss_list.append(loss)
        train_list.append(train_MAE * std)
        val_list.append(val_MAE)
        plot_loss(train_list, val_list, "./training_curve.png")
        lr_list.append(lr)
        if train["early_stopping"]:
            if early_stopper.stop(val_MAE, epoch):
                print("Early stopping at epoch {}.".format(epoch))
                break
    print("-----------------------------------------------------------------------------------------")
    training_time = (time.time() - t0) / 60
    print("Training time: {:.2f} min".format(training_time))
    device_dict["training_time"] = training_time
    # Generate report of the training process
    create_model_report(
        output_name,
        output_directory,
        hyperparameters,
        model,
        (train_loader, val_loader, test_loader),
        (mean, std),
        (train_list, val_list, test_list, lr_list),
        device_dict,
    )

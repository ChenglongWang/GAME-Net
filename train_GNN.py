"""Launch GNN model training."""

__author__ = "Santiago Morandi"

import argparse
from os.path import exists, isdir
import sys
import time
import os

import torch 
from torch_geometric.nn import GraphMultisetTransformer
import toml

from functions import create_loaders, scale_target, train_loop, test_loop
from nets import FlexibleNet
from post_training import create_model_report
from create_graph_datasets import create_graph_datasets
from constants import loss_dict, pool_seq_dict, conv_layer, sigma_dict, pool_dict
from paths import pre_id

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform a training process with the hyperparameter setting defined in train.toml")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Name of the training.")
    ARGS = PARSER.parse_args()
    
    if isdir("./Models/{}".format(ARGS.o)):
        ARGS.o = input("There is already a model with the provided name, provide a new one: ")
    
    # Upload training hyperparameters from config.toml
    HYPERPARAMS = toml.load("hyper_config.toml")  
    data = HYPERPARAMS["data"]["root"]    
    graph = HYPERPARAMS["graph"]
    train = HYPERPARAMS["train"]
    architecture = HYPERPARAMS["architecture"]
        
    # Select device for the learning process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current device: {}".format(device))
    if device == "cuda":
        print("Device name: {}".format(torch.cuda.get_device_name(0)))
        print("CudaDNN enabled: {}".format(torch.backends.cudnn.enabled)) 
        print("CUDA Version: {}".format(torch.version.cuda))
        print("CuDNN Version: {}".format(torch.backends.cudnn.version()))
        device_dict = {"name": torch.cuda.get_device_name(0)}
    else:
        print("Training with CPU")
        device_dict = {"name": "CPU"}
     
    # Convert raw DFT data to graph representation or upload existing setting  
    if exists(data + "/amides/" + pre_id):  
        from processed_datasets import FG_dataset
    else:
        create_graph_datasets(graph['voronoi_tol'], 
                              graph['second_order_nn'], 
                              graph['scaling_factor'])
        from processed_datasets import FG_dataset
    
    # Create train, validation and test sets Dataloaders  
    train_loader, val_loader, test_loader = create_loaders(FG_dataset,
                                                           batch_size=train["batch_size"],
                                                           split=train["splits"], 
                                                           test=train["test_set"])    
    # Apply target scaling 
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=train["target_scaling"], 
                                                                    test=train["test_set"])    
    # Instantiate the GNN model and move it on the selected device
    model = FlexibleNet(dim=architecture["dim"],
                        N_linear=architecture["n_linear"], 
                        N_conv=architecture["n_conv"], 
                        adj_conv=architecture["adj_conv"],  
                        sigma=sigma_dict[architecture["sigma"]], 
                        bias=architecture["bias"], 
                        conv=conv_layer[architecture["conv_layer"]], 
                        pool=pool_dict[architecture["pool_layer"]], 
                        pool_ratio=architecture["pool_ratio"], 
                        pool_heads=architecture["pool_heads"], 
                        pool_seq=pool_seq_dict[architecture["pool_seq"]], 
                        pool_layer_norm=architecture["pool_layer_norm"]).to(device)     
    # Load optimizer and LR scheduler for steering the learning process
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train["lr0"],
                                 eps=train["eps"], 
                                 weight_decay=train["weight_decay"],
                                 amsgrad=train["amsgrad"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=train["factor"],
                                                              patience=train["patience"],
                                                              min_lr=train["minlr"])    
    # Run the learning process    
    loss_list = []  # Loss function during training
    train_list = [] # Training MAE during training
    val_list = []   # Validation MAE during training
    test_list = []  # Test MAE during training     
    t0 = time.time()
    for epoch in range(1, train["epochs"]+1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]['lr']        
        # Run epoch over training set
        loss, train_MAE = train_loop(model, device, train_loader, optimizer, loss_dict[train["loss_function"]])  
        # Run epoch over validation set
        val_MAE = test_loop(model, val_loader, device, std)  
        # Adjust lr based on validation error                             
        lr_scheduler.step(val_MAE)
        # Run epoch over test set                                                                            
        #test_MAE = test_loop(model, test_loader, device, std)                             
        
        if train["test_set"]:
            test_MAE = test_loop(model, test_loader, device, std, mean)         
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                  'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
            test_list.append(test_MAE)
        else:
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV '
                  .format(epoch, lr, train_MAE*std, val_MAE))
         
        loss_list.append(loss)
        train_list.append(train_MAE * std)
        val_list.append(val_MAE)
    print("-----------------------------------------------------------------------------------------")
    training_time = (time.time() - t0)/60  # in minutes
    print("Training time: {:.2f} min".format(training_time))
    device_dict["training_time"] = training_time
    create_model_report(ARGS.o,
                        HYPERPARAMS,  
                        model, 
                        (train_loader, val_loader, test_loader),
                        (mean, std),
                        (train_list, val_list, test_list), 
                        device_dict)
    
    
    
    
    
    

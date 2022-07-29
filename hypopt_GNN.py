"""Run Hyperparameter optimization with Ray Tune"""

__author__ = "Santiago Morandi"

import argparse
import sys
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from ray import tune
import ray

from nets import SantyxNet
from functions import train_loop, test_loop, scale_target, create_loaders
from processed_datasets import FG_dataset, BM_dataloader

HYPERPARAMS = {}
# Process-related
HYPERPARAMS["test set"] = True          
HYPERPARAMS["splits"] = 10              
HYPERPARAMS["target scaling"] = "std"   
HYPERPARAMS["batch size"] = tune.choice([16, 32, 64])           
HYPERPARAMS["epochs"] = 200               
HYPERPARAMS["loss function"] = torch.nn.functional.l1_loss   
HYPERPARAMS["lr0"] = tune.choice([0.01, 0.001, 0.0001])       
HYPERPARAMS["patience"] = tune.choice([5, 7, 10])              
HYPERPARAMS["factor"] = tune.choice([0.5, 0.7, 0.9])          
HYPERPARAMS["minlr"] = tune.choice([1e-7, 1e-8])             
HYPERPARAMS["betas"] = (0.9, 0.999)     
HYPERPARAMS["eps"] = tune.choice([1e-8, 1e-9])               
HYPERPARAMS["weight decay"] = 0         
HYPERPARAMS["amsgrad"] = tune.choice([True, False])       
# Model-related
HYPERPARAMS["dim"] = tune.choice([64, 128, 256])                
HYPERPARAMS["sigma"] = torch.nn.ReLU()  
HYPERPARAMS["bias"] = tune.choice([True, False])              
HYPERPARAMS["conv normalize"] = False   
HYPERPARAMS["conv root weight"] = True
HYPERPARAMS["pool ratio"] = tune.choice([0.25, 0.5, 0.75])        
HYPERPARAMS["pool heads"] = tune.choice([2, 4, 6])
HYPERPARAMS["pool seq"] = tune.choice([["GMPool_I"], 
                                       ["GMPool_G"], 
                                       ["GMPool_G", "GMPool_I"],
                                       ["GMPool_G", "SelfAtt", "GMPool_I"],
                                       ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]])
HYPERPARAMS["pool layer norm"] = False 

def train_function(config, checkpoint_dir=None):
    """
    Function for hyperparameter tuning via RayTune.
    Args:
        config (dict): Dictionary with search space (hyperparameters)
    """    
    # Generate Datasets and scale target
    train_loader, val_loader, test_loader = create_loaders(FG_dataset,
                                                           config["splits"],
                                                           config["batch size"], 
                                                           config["test set"])
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=config["target scaling"], 
                                                                    test=config["test set"])    
    # Select device 
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    # Call GNN model 
    model = SantyxNet(dim=config["dim"],
                      sigma=config["sigma"], 
                      bias=config["bias"], 
                      conv_normalize=config["conv normalize"], 
                      conv_root_weight=config["conv root weight"], 
                      pool_ratio=config["pool ratio"], 
                      pool_heads=config["pool heads"], 
                      pool_seq=config["pool seq"], 
                      pool_layer_norm=config["pool layer norm"]).to(device)    
    # Call optimizer and lr-scheduler
    optimizer = Adam(model.parameters(),
                     lr=config["lr0"], 
                     betas=config["betas"],
                     eps=config["eps"], 
                     weight_decay=config["weight decay"], 
                     amsgrad=config["amsgrad"])
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min',
                                     factor=config["factor"],
                                     patience=config["patience"],
                                     min_lr=config["minlr"])    
    # Run training
    for epoch in range(1, config["epochs"]+1):
        lr = lr_scheduler.optimizer.param_groups[0]['lr']
        _, train_MAE = train_loop(model, device, train_loader, optimizer, config["loss function"])  
        val_MAE = test_loop(model, val_loader, device, std)
        lr_scheduler.step(val_MAE)  
        if config["test set"]:
            test_MAE = test_loop(model, test_loader, device, std)                                           
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                  'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
        else:
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV '
                  .format(epoch, lr, train_MAE*std, val_MAE))      
    # Collect performance metric
    BM_MAE = test_loop(model, BM_dataloader, device=device, std=std, mean=mean, scaled_graph_label=False)
    FG_MAE = test_MAE             
    tune.report(BM_MAE=BM_MAE, FG_MAE=FG_MAE)
    
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Run hyperparameter optimization for the GNN model.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Name of the folder where results will be stored.")
    PARSER.add_argument("-s", "--samples", default=10, type=int, dest="s",
                        help="Number of samples during the search.")
    PARSER.add_argument("-v", "--verbose", default=1, type=int, dest="v",
                        help="Verbosity of tune.run() function")
    ARGS = PARSER.parse_args()
    
    ray.init(ignore_reinit_error=True)
    analysis = tune.run(train_function,
                        metric="BM_MAE",
                        mode="min",
                        name=ARGS.o,
                        time_budget_s=3600*24,
                        config=HYPERPARAMS,
                        #scheduler=ASHAScheduler,
                        #checkpoint_freq=5,
                        #progress_reporter=CLIReporter,
                        resources_per_trial={"cpu":8, "gpu":1},
                        num_samples=ARGS.s, 
                        verbose=ARGS.v,
                        log_to_file=True, 
                        local_dir="./Hyperparameter_Optimization")

    
"""Run Hyperparameter optimization with Ray Tune"""

import argparse
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/gnn_eads')))
os.environ["TUNE_RESULT_DELIM"] = "/"
from gnn_eads.paths import create_paths
from gnn_eads.create_graph_datasets import create_graph_datasets
from gnn_eads.processed_datasets import create_post_processed_datasets
from gnn_eads.constants import NODE_FEATURES, FG_RAW_GROUPS
from gnn_eads.functions import get_id, create_loaders, scale_target, get_id, train_loop, test_loop
from gnn_eads.nets import FlexibleNet

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphMultisetTransformer
from torch.nn import ReLU, Tanh, Sigmoid  
from torch.optim import Adam
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import pandas as pd
#--------------------------------------------------------------------------------------------#
HYPERPARAMS = {}
# Graph settings (THESE ARE NOT HYPERPARAMS, DO NOT SWITCH THEM TO VARIABLES!)
HYPERPARAMS["voronoi_tol"] = 0.5   
HYPERPARAMS["scaling_factor"] = 1.5
HYPERPARAMS["second_order_nn"] = False
HYPERPARAMS["data"] = "/home/santiago/Desktop/GNN/data/FG_dataset"
# Process-related
HYPERPARAMS["test_set"] = True          
HYPERPARAMS["splits"] = 10              
HYPERPARAMS["target_scaling"] = "std"   
HYPERPARAMS["batch_size"] = tune.choice([16, 32, 64])           
HYPERPARAMS["epochs"] = 200               
HYPERPARAMS["loss_function"] = torch.nn.functional.l1_loss   
HYPERPARAMS["lr0"] = tune.choice([0.001, 0.0001])       
HYPERPARAMS["patience"] = tune.choice([5, 7])              
HYPERPARAMS["factor"] = tune.choice([0.5, 0.7, 0.9])          
HYPERPARAMS["minlr"] = tune.choice([1e-8])             
HYPERPARAMS["betas"] = (0.9, 0.999)     
HYPERPARAMS["eps"] = tune.choice([1e-8, 1e-9])               
HYPERPARAMS["weight_decay"] = 0         
HYPERPARAMS["amsgrad"] = tune.choice([True, False])       
# Model-related
HYPERPARAMS["dim"] = tune.choice([128, 256])                
HYPERPARAMS["sigma"] = tune.choice([torch.nn.ReLU()])  
HYPERPARAMS["bias"] = tune.choice([True, False])  
HYPERPARAMS["adj_conv"] = tune.choice([True, False])  
HYPERPARAMS["n_linear"] = tune.choice([0, 1, 3, 5]) 
HYPERPARAMS["n_conv"] = tune.choice([3, 4, 5]) 
HYPERPARAMS["conv_layer"] = tune.choice([SAGEConv, GATv2Conv]) 
HYPERPARAMS["pool_layer"] =  tune.choice([GraphMultisetTransformer])           
HYPERPARAMS["conv_normalize"] = False   
HYPERPARAMS["conv_root_weight"] = True
HYPERPARAMS["pool_ratio"] = tune.choice([0.25, 0.5, 0.75])        
HYPERPARAMS["pool_heads"] = tune.choice([1, 2, 4])
HYPERPARAMS["pool_seq"] = tune.choice([["GMPool_I"], 
                                       ["GMPool_G"], 
                                       ["GMPool_G", "GMPool_I"],
                                       ["GMPool_G", "SelfAtt", "GMPool_I"],
                                       ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]])
HYPERPARAMS["pool_layer_norm"] = False 
# TODO: count the max number of possible combinations of the defined hyperparameter space
#--------------------------------------------------------------------------------------------#
PARSER = argparse.ArgumentParser(description="Perform hyperparameter optimization with Ray-Tune (ASHA algorithm). \
                                 The output is stored in the \"hyperparameter_optimization\" directory.")
PARSER.add_argument("-o", "--output", type=str, dest="o", 
                    help="Name of the hyperparameter optimization run.")
PARSER.add_argument("-s", "--samples", default=5, type=int, dest="s",
                    help="Number of trials of the search.")
PARSER.add_argument("-v", "--verbose", default=1, type=int, dest="v",
                    help="Verbosity of tune.run() function")
PARSER.add_argument("-gr", "--grace", default=15, type=int, dest="grace", 
                    help="Grace period of ASHA.")
PARSER.add_argument("-maxit", "--max-iterations", default=150, type=int, dest="max_iter", 
                    help="Maximum number of training iterations (epochs) allowed by ASHA.")
PARSER.add_argument("-rf", "--reduction-factor", default=4, type=int, dest="rf", 
                    help="Reduction factor of ASHA.")
PARSER.add_argument("-bra", "--brackets", default=1, type=int, dest="bra", 
                    help="Number of brackets of ASHA.")
PARSER.add_argument("-gpt", "--gpu-per-trial", default=1.0, type=float, dest="gpu_per_trial", 
                    help="Number of gpus per trial (can be fractional).")    
ARGS = PARSER.parse_args()
    
def train_function(config: dict, checkpoint_dir=None):
    """
    Function for hyperparameter tuning via RayTune.
    Args:
        config (dict): Dictionary with search space (hyperparameters)
    """ 
    data_path = HYPERPARAMS["data"]
    graph_identifier = get_id(config)
    family_paths = create_paths(FG_RAW_GROUPS, data_path, graph_identifier)
    if os.path.exists(data_path + "/amides/pre_" + graph_identifier):  
        FG_dataset = create_post_processed_datasets(graph_identifier, family_paths)
    else:
        print("Creating graphs from raw data ...")  
        create_graph_datasets(config, family_paths)
        FG_dataset = create_post_processed_datasets(graph_identifier, family_paths) 
    train_loader, val_loader, test_loader = create_loaders(FG_dataset,
                                                           config["splits"],
                                                           config["batch_size"], 
                                                           config["test_set"])
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=config["target_scaling"], 
                                                                    test=config["test_set"])    
    device = "cuda" if torch.cuda.is_available() else "cpu"            
    model = FlexibleNet(dim=config["dim"],
                        N_linear=config["n_linear"], 
                        N_conv=config["n_conv"], 
                        adj_conv=config["adj_conv"], 
                        in_features=NODE_FEATURES, 
                        sigma=config["sigma"], 
                        bias=config["bias"], 
                        conv=config["conv_layer"], 
                        pool=config["pool_layer"], 
                        pool_ratio=config["pool_ratio"], 
                        pool_heads=config["pool_heads"], 
                        pool_seq=config["pool_seq"], 
                        pool_layer_norm=config["pool_layer_norm"]).to(device)     
    optimizer = Adam(model.parameters(),
                     lr=config["lr0"], 
                     betas=config["betas"],
                     eps=config["eps"], 
                     weight_decay=config["weight_decay"], 
                     amsgrad=config["amsgrad"])
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min',
                                     factor=config["factor"],
                                     patience=config["patience"],
                                     min_lr=config["minlr"])    
    train_list, val_list, test_list = [], [], [] # MAE vs epoch
    for iteration in range(1, config["epochs"]+1):
        lr = lr_scheduler.optimizer.param_groups[0]['lr']
        _, train_MAE = train_loop(model, device, train_loader, optimizer, config["loss_function"])  
        val_MAE = test_loop(model, val_loader, device, std)
        lr_scheduler.step(val_MAE)  
        if config["test_set"]:
            test_MAE = test_loop(model, test_loader, device, std)                                           
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                  'Test MAE: {:.4f} eV'.format(iteration, lr, train_MAE*std, val_MAE, test_MAE))
        else:
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV '
                  .format(iteration, lr, train_MAE*std, val_MAE))      
        # BM_MAE = test_loop(model, BM_dataloader, device=device, std=std, mean=mean, scaled_graph_label=False)                    
        #tune.report(BM_MAE=BM_MAE, FG_MAE=test_MAE, epoch=iteration)            
        train_list.append(train_MAE * std)
        val_list.append(val_MAE)
        test_list.append(test_MAE)
        tune.report(FG_MAE=test_MAE, epoch=iteration)  
    # if BM_MAE <= ARGS.target:  # If the extrapolation performance is good, save the model!
    #     time = str(datetime.now())[11:]
    #     time = time[:8]
    #     create_model_report("{}_{}".format(ARGS.o, time),
    #                         HYPERPARAMS,  # Wrong
    #                         model,
    #                         (train_loader, val_loader, test_loader),
    #                         (mean, std),
    #                         (train_list, val_list, test_list))
    
hypopt_scheduler = ASHAScheduler(time_attr="epoch",
                                 metric="FG_MAE",
                                 mode="min",
                                 grace_period=ARGS.grace,
                                 reduction_factor=ARGS.rf,
                                 max_t=ARGS.max_iter,
                                 brackets=ARGS.bra)

def main():
    ray.init(ignore_reinit_error=True)
    result = tune.run(train_function,
                      name=ARGS.o,
                      time_budget_s=3600*24,
                      config=HYPERPARAMS,
                      scheduler=hypopt_scheduler,
                      resources_per_trial={"cpu":1, "gpu":ARGS.gpu_per_trial},
                      num_samples=ARGS.s, 
                      verbose=ARGS.v,
                      log_to_file=True, 
                      local_dir="../hyperparameter_optimization",
                      raise_on_failed_trial=False)
    ray.shutdown()  
    best_config = result.get_best_config(metric="FG_MAE", mode="min", scope="last")
    best_config_df = pd.DataFrame.from_dict(best_config, orient="index")    
    best_config_df.to_csv("../hyperparameter_optimization/{}/best_config.csv".format(ARGS.o), sep="/")
    print(best_config)
    exp_df = result.results_df
    exp_df.to_csv("../hyperparameter_optimization/{}/summary.csv".format(ARGS.o), sep="/")  

if __name__ == "__main__":  
    main()   
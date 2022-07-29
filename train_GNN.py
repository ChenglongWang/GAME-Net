"""Run a GNN learning process with the provided model and hyperparameters."""

__author__ = "Santiago Morandi"

import argparse
import sys
import time

import torch 
import torch_geometric
from torch.nn.functional import l1_loss, mse_loss, huber_loss

from functions import create_loaders, scale_target, train_loop, test_loop
from processed_datasets import FG_dataset
from nets import SantyxNet
from post_training import create_model_report

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Run a training of the GNN model.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Name of the folder where results will be stored.")
    PARSER.add_argument("-d", "--dimension", default=128, type=int, dest="d",
                        help="Dimension of each layer size defined in the GNN model structure.")
    PARSER.add_argument("-b", "--batchsize", default=32, type=int, dest="b", 
                        help="Batch size.")
    PARSER.add_argument("-s", "--split", default=10, type=int, dest="s",
                        help="Number of splits for creating training/validation/test sets.")
    PARSER.add_argument("-e", "--epochs", default=300, type=int, dest="e",
                    help="Number of epochs performed in the training.")
    PARSER.add_argument("-lr", "--learning_rate", default=0.001, type=float, dest="lr",
                    help="Initial learning rate of the training.")
    PARSER.add_argument("-minlr", "--min_learning_rate", default=1e-6, type=float, dest="minlr",
                    help="Minimum learning rate of the training.")
    PARSER.add_argument("-p", "--patience", default=5, type=int, dest="p",
                    help="Patience of the learning rate scheduler.")
    PARSER.add_argument("-f", "--factor", default=0.7, type=float, dest="f",
                    help="Decreasing factor for the learning rate scheduler.")
    PARSER.add_argument("-j", "--loss_function", default="mae", type=str, dest="j",
                    help="Loss function minimized in the learning process.", choices=["mae", "mse", "huber"])
    PARSER.add_argument("-sigma", default="relu", type=str, dest="sigma", 
                        help="Activation function of the model", choices=["relu", "tanh", "sigmoid"])
    
    ARGS = PARSER.parse_args()
    
    if ARGS.j == "mse":
        LOSS_FUNCTION = mse_loss
    elif ARGS.j == "mae":
        LOSS_FUNCTION = l1_loss
    elif ARGS.j == "huber":
        LOSS_FUNCTION = huber_loss
    else:
        LOSS_FUNCTION = None    
    OUTPUT_NAME = ARGS.o
    
    if ARGS.sigma == "relu":
        SIGMA = torch.nn.ReLU()
    elif ARGS.sigma == "tanh":
        SIGMA = torch.nn.Tanh()
    elif ARGS.sigma == "sigmoid":
        SIGMA = torch.nn.Sigmoid()
    else:
        SIGMA = None
    
    # Select device for the learning process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current device: {}".format(device))
    if device == "cuda":
        print("Device name: {}".format(torch.cuda.get_device_name(0)))
        print("CudaDNN enabled: {}".format(torch.backends.cudnn.enabled)) 
        print("CUDA Version: {}".format(torch.version.cuda))
        print("CuDNN Version: {}".format(torch.backends.cudnn.version()))
    print("Python version: {}".format(sys.version[:7]))
    print("Pytorch version: {}".format(torch.__version__))
    print("Pytorch Geometric version: {}".format(torch_geometric.__version__))
    
    HYPERPARAMS = {}
    # Process-related
    HYPERPARAMS["splits"] = ARGS.s          # Splits of the FG-dataset for train-val-test sets creation
    HYPERPARAMS["target scaling"] = "std"   # Target scaling (standardization)
    HYPERPARAMS["test set"] = True          # True=Generate train-val-test  False=Generate train-val (whole FG-dataset)
    HYPERPARAMS["batch size"] = ARGS.b           
    HYPERPARAMS["epochs"] = ARGS.e               
    HYPERPARAMS["loss function"] = LOSS_FUNCTION
    # Learning rate scheduler   
    HYPERPARAMS["lr0"] = ARGS.lr             
    HYPERPARAMS["patience"] = ARGS.p          
    HYPERPARAMS["factor"] = ARGS.f           
    HYPERPARAMS["minlr"] = ARGS.minlr
    # Optimizer (Adam)             
    HYPERPARAMS["betas"] = (0.9, 0.999)      
    HYPERPARAMS["eps"] = 1e-8                
    HYPERPARAMS["weight decay"] = 0          
    HYPERPARAMS["amsgrad"] = False               
    # Global model architecture
    HYPERPARAMS["dim"] = ARGS.d              
    HYPERPARAMS["sigma"] = SIGMA                     
    HYPERPARAMS["bias"] = True               
    # GraphSAGE (convolution)
    HYPERPARAMS["conv normalize"] = False    
    HYPERPARAMS["conv root weight"] = True    
    HYPERPARAMS["pool ratio"] = 0.25         
    # GraphMultiset Transformer (pooling)
    HYPERPARAMS["pool heads"] = 4
    HYPERPARAMS["pool seq"] = ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]
    HYPERPARAMS["pool layer norm"] = False 
    
    # 1) Create train/validation/test sets   
    train_loader, val_loader, test_loader = create_loaders(FG_dataset,
                                                           batch_size=HYPERPARAMS["batch size"],
                                                           split=HYPERPARAMS["splits"], 
                                                           test=HYPERPARAMS["test set"])    
    # 2) Apply target scaling 
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=HYPERPARAMS["target scaling"], 
                                                                    test=HYPERPARAMS["test set"])    
    # 3) Instantiate the model and move it on the selected device
    model = SantyxNet(dim=HYPERPARAMS["dim"],
                      sigma=HYPERPARAMS["sigma"], 
                      bias=HYPERPARAMS["bias"], 
                      conv_normalize=HYPERPARAMS["conv normalize"], 
                      conv_root_weight=HYPERPARAMS["conv root weight"], 
                      pool_ratio=HYPERPARAMS["pool ratio"], 
                      pool_heads=HYPERPARAMS["pool heads"], 
                      pool_seq=HYPERPARAMS["pool seq"], 
                      pool_layer_norm=HYPERPARAMS["pool layer norm"]).to(device)     
    
    # 4) Load optimizer+lr-scheduler for steering the training
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HYPERPARAMS["lr0"],
                                 betas=HYPERPARAMS["betas"], 
                                 eps=HYPERPARAMS["eps"], 
                                 weight_decay=HYPERPARAMS["weight decay"],
                                 amsgrad=HYPERPARAMS["amsgrad"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=HYPERPARAMS["factor"],
                                                              patience=HYPERPARAMS["patience"],
                                                              min_lr=HYPERPARAMS["minlr"])    
    # 5) Run the training phase    
    best_val_error = None
    loss_list = []  
    train_list = [] 
    val_list = []   
    test_list = []  
     
    t0 = time.time()
    for epoch in range(1, HYPERPARAMS["epochs"]+1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]['lr']        
        # Run epoch over training set
        loss, train_MAE = train_loop(model, device, train_loader, optimizer, HYPERPARAMS["loss function"])  
        # Run epoch over validation set
        val_MAE = test_loop(model, val_loader, device, std)  
        # Adjust lr based on validation error                             
        lr_scheduler.step(val_MAE)
        # Run epoch over test set                                                                            
        test_MAE = test_loop(model, test_loader, device, std)                             
    
        if LOSS_FUNCTION == l1_loss:
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV,  Validation MAE: {:.4f} eV, '
              'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
        else:
            print('Epoch {:03d}: LR={:.7f}  Loss={:.6f}  Validation MAE: {:.6f} eV, '
                  'Test MAE: {:.6f} eV'.format(epoch, lr, loss, val_MAE, test_MAE))     
        
        loss_list.append(loss)
        train_list.append(train_MAE * std)
        val_list.append(val_MAE)
        test_list.append(test_MAE)
        
        # fig, ax = E_violinplot_train(model, test_loader, std_tv, set(FG_FAMILIES), epoch)
        # plt.savefig("./gif/violin_{}.png".format(epoch), dpi=200, bbox_inches='tight')
        # fig, ax = DFTvsGNN_plot_train(model, test_loader, mean_tv, std_tv, epoch)
        # plt.savefig("./gif/parity_{}.png".format(epoch), dpi=200, bbox_inches='tight')
    print("-----------------------------------------------------------------------------------------")
    print("device: {}    Training time: {:.2f} s".format(device, time.time() - t0))
    create_model_report(OUTPUT_NAME, 
                        model, 
                        (train_loader, val_loader, test_loader),
                        (mean, std),
                        HYPERPARAMS, 
                        (train_list, val_list, test_list))
    
    
    
    
    
    

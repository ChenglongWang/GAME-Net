"""Script for launching a GNN model training."""

__author__ = "Santiago Morandi"

import argparse
import sys
import time

import torch 
import torch_geometric
from torch.nn.functional import l1_loss, mse_loss, huber_loss
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphMultisetTransformer

from functions import create_loaders, scale_target, train_loop, test_loop
from processed_datasets import FG_dataset
from nets import FlexibleNet
from post_training import create_model_report
from constants import NODE_FEATURES

# Possible hyperparameters for loss function, convolutional layer and GMT pool sequence
loss_dict = {"mse": mse_loss, "mae": l1_loss, "huber": huber_loss}
pool_seq_dict = {"1": ["GMPool_I"],
                 "2": ["GMPool_G"],
                 "3": ["GMPool_G", "GMPool_I"],
                 "4": ["GMPool_G", "SelfAtt", "GMPool_I"], 
                 "5": ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]}
conv_layer = {"SAGE": SAGEConv, "GATv2": GATv2Conv}

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform a single training process with the selected hyperparameter setting.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Name of the directory where results will be stored.")
    PARSER.add_argument("-d", "--dimension", default=128, type=int, dest="d",
                        help="Dimension of each layer defined in the GNN model structure.")
    PARSER.add_argument("-b", "--batchsize", default=32, type=int, dest="b", 
                        help="Batch size.")
    PARSER.add_argument("-s", "--split", default=10, type=int, dest="s",
                        help="Number of splits for creating train/val/test sets.")
    PARSER.add_argument("-e", "--epochs", default=300, type=int, dest="e",
                    help="Epochs performed in the training.")
    PARSER.add_argument("-lr", "--learning_rate", default=0.001, type=float, dest="lr",
                    help="Initial learning rate.")
    PARSER.add_argument("-minlr", "--min_learning_rate", default=1e-6, type=float, dest="minlr",
                    help="Minimum learning rate.")
    PARSER.add_argument("-p", "--patience", default=5, type=int, dest="p",
                    help="Patience of the learning rate scheduler.")
    PARSER.add_argument("-f", "--factor", default=0.7, type=float, dest="f",
                    help="Decreasing factor for the learning rate scheduler.")
    PARSER.add_argument("-j", "--loss_function", default="mae", type=str, dest="loss",
                    help="Loss function of the training.")
    PARSER.add_argument("-ac", "--adj_conv", default=True, type=bool, dest="adj_conv",
                    help="Whether coupling convolutional layer with fully connected layer.")
    PARSER.add_argument("-nl", "--n_linear", default=1, type=int, dest="n_linear",
                    help="Number of linear layers after input layer.")
    PARSER.add_argument("-nc", "--n_convolutions", default=3, type=int, dest="n_conv",
                    help="Number of convolutional layers.")
    PARSER.add_argument("-conv", "--convolutional_layer", default="SAGE", type=str, dest="conv_layer",
                    help="Type of Convolutional layer applied.")
    PARSER.add_argument("-pr", "--pool_ratio", default=0.25, type=float, dest="pool_ratio",
                    help="Pooling ratio applied by the GMT Transformer.")
    PARSER.add_argument("-ph", "--pool_heads", default=2, type=int, dest="pool_heads",
                    help="Number of pooling heads in the GMT Transformer.")
    PARSER.add_argument("-ps", "--pool_seq", default="1", type=str, dest="pool_seq",
                    help="Pooling sequence applied by the GMT Transformer.")
    PARSER.add_argument("-pln", "--pool_layer_norm", default=False, type=bool, dest="pool_layer_norm",
                    help="Whether applying layer normalization in the GMT Transormer.")
    PARSER.add_argument("-bias", default=True, type=bool, dest="bias", help="Whether allowing bias in all the GNN layers.")
    
    ARGS = PARSER.parse_args()
    
    LOSS_FUNCTION = loss_dict[ARGS.loss]    
    OUTPUT_NAME = ARGS.o
    CONV_LAYER = conv_layer[ARGS.conv_layer]
    POOL_SEQ = pool_seq_dict[ARGS.pool_seq]
    
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
    HYPERPARAMS["splits"] = ARGS.s          # Splits among which the data are partitioned to create train-val-test sets
    HYPERPARAMS["target_scaling"] = "std"   # Target scaling approach (std=standardization)
    HYPERPARAMS["test_set"] = True          # True=Generate train-val-test sets. False=Generate train-val (train with whole FG-dataset)
    HYPERPARAMS["batch_size"] = ARGS.b           
    HYPERPARAMS["epochs"] = ARGS.e               
    HYPERPARAMS["loss_function"] = LOSS_FUNCTION   
    HYPERPARAMS["lr0"] = ARGS.lr             # Initial learning rate (lr)
    HYPERPARAMS["patience"] = ARGS.p         # Lr scheduler patience 
    HYPERPARAMS["factor"] = ARGS.f           # Decreasing factor of the lr scheduler
    HYPERPARAMS["minlr"] = ARGS.minlr             
    HYPERPARAMS["betas"] = (0.9, 0.999)      # Adam optimizer: betas
    HYPERPARAMS["eps"] = 1e-8                # Adam optimizer: eps
    HYPERPARAMS["weight_decay"] = 0          # Adam optimizer: weight decay
    HYPERPARAMS["amsgrad"] = True            # Adam optimizer: amsgrad    
    # Model-related
    HYPERPARAMS["dim"] = ARGS.d              # Depth of the GNN layers
    HYPERPARAMS["sigma"] = torch.nn.ReLU()   # Activation function of the model       
    HYPERPARAMS["bias"] = ARGS.bias  
    HYPERPARAMS["n_linear"] = ARGS.n_linear 
    HYPERPARAMS["n_conv"] = ARGS.n_conv 
    HYPERPARAMS["conv_layer"] = CONV_LAYER
    HYPERPARAMS["adj_conv"] = ARGS.adj_conv     
    HYPERPARAMS["conv_normalize"] = False   
    HYPERPARAMS["conv_root_weight"] = True
    HYPERPARAMS["pool_layer"] =  GraphMultisetTransformer
    HYPERPARAMS["pool_ratio"] = ARGS.pool_ratio        
    HYPERPARAMS["pool_heads"] = ARGS.pool_heads
    HYPERPARAMS["pool_seq"] = POOL_SEQ
    HYPERPARAMS["pool_layer_norm"] = ARGS.pool_layer_norm
 
    # Create train, validation and test sets Dataloaders  
    train_loader, val_loader, test_loader = create_loaders(FG_dataset,
                                                           batch_size=HYPERPARAMS["batch_size"],
                                                           split=HYPERPARAMS["splits"], 
                                                           test=HYPERPARAMS["test_set"])    
    # Apply target scaling 
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=HYPERPARAMS["target_scaling"], 
                                                                    test=HYPERPARAMS["test_set"])    
    # Instantiate the GNN model and move it on the available device
    model = FlexibleNet(dim=HYPERPARAMS["dim"],
                        N_linear=HYPERPARAMS["n_linear"], 
                        N_conv=HYPERPARAMS["n_conv"], 
                        adj_conv=HYPERPARAMS["adj_conv"], 
                        in_features=NODE_FEATURES, 
                        sigma=HYPERPARAMS["sigma"], 
                        bias=HYPERPARAMS["bias"], 
                        conv=HYPERPARAMS["conv_layer"], 
                        pool=HYPERPARAMS["pool_layer"], 
                        pool_ratio=HYPERPARAMS["pool_ratio"], 
                        pool_heads=HYPERPARAMS["pool_heads"], 
                        pool_seq=HYPERPARAMS["pool_seq"], 
                        pool_layer_norm=HYPERPARAMS["pool_layer_norm"]).to(device) 
    
    # Load optimizer and LR scheduler for steering the learning process
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HYPERPARAMS["lr0"],
                                 betas=HYPERPARAMS["betas"], 
                                 eps=HYPERPARAMS["eps"], 
                                 weight_decay=HYPERPARAMS["weight_decay"],
                                 amsgrad=HYPERPARAMS["amsgrad"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=HYPERPARAMS["factor"],
                                                              patience=HYPERPARAMS["patience"],
                                                              min_lr=HYPERPARAMS["minlr"])
    
    # Run the learning process    
    best_val_error = None
    loss_list = []  # Store loss function trend during training
    train_list = [] # Store training MAE during training
    val_list = []   # Store validation MAE during training
    test_list = []  # Store test MAE during training
     
    t0 = time.time()
    for epoch in range(1, HYPERPARAMS["epochs"]+1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]['lr']        
        # Run epoch over training set
        loss, train_MAE = train_loop(model, device, train_loader, optimizer, HYPERPARAMS["loss_function"])  
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
    print("Training time: {:.2f} s".format(time.time() - t0))
    print("Device: {}".format(torch.cuda.get_device_name(0)))

    create_model_report(OUTPUT_NAME, 
                        model, 
                        (train_loader, val_loader, test_loader),
                        (mean, std),
                        HYPERPARAMS, 
                        (train_list, val_list, test_list))
    
    
    
    
    
    

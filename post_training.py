"""
Module for post-processing and collecting data.
"""

import os
from datetime import date, datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F
from torchinfo import summary

from constants import ENCODER, FG_FAMILIES, DPI
from functions import get_graph_formula, get_number_atoms_from_label, split_percentage
from graph_tools import plotter
from plot_functions import hist_num_atoms, violinplot_family, DFTvsGNN_plot, pred_real, training_plot

def create_model_report(model_name: str,
                        model,
                        loaders: tuple[DataLoader],                     
                        scaling_params : tuple[float], 
                        hyperparams: dict,
                        mae_lists: tuple[list], 
                        device: dict=None):
    """
    Create report of the performed GNN learning process.

    Args:
        model_name (str): Output name of the model
        model (_type_): GNN model object
        loaders (tuple[DataLoader]): Train, val and test DataLoaders
        scaling_params (tuple[float]): Scaling parameters
        hyperparams (dict): Dictionary with the hyperparameters of the learning process
        mae_lists (tuple[list]): MAE of the train/val/test sets wrt epoch
    """
    # Report day and time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)
    
    
    # Unfold parameters
    train_loader = loaders[0]
    val_loader = loaders[1]
    test_loader = loaders[2]
    
    # Scaling parameters
    if hyperparams["target scaling"] == "std":
        mean_tv = scaling_params[0]
        std_tv = scaling_params[1]
    elif hyperparams["target scaling"] == "norm":
        pass #TODO
    else:
        pass
    
    # Unfold lists of MAE vs epoch
    train_list = mae_lists[0]
    val_list = mae_lists[1]
    test_list = mae_lists[2]
    
    # Create folder where to store files
    try:
        os.mkdir("./Models/{}".format(model_name))
    except FileExistsError:
        model_name = input("The name provided already exists: Provide a new one: ")
        os.mkdir("./Models/{}".format(model_name))
    os.mkdir("./Models/{}/Outliers".format(model_name))
    
    # Store info about GNN architecture 
    with open('./Models/{}/architecture.txt'.format(model_name), 'w') as f:
        print(summary(model, batch_dim=hyperparams["batch size"], verbose=2), file=f)
    
    # Save GNN model object
    torch.save(model, "./Models/{}/model.pth".format(model_name))
    # Save GNN model parameters 
    torch.save(model.state_dict(), "./Models/{}/GNN.pth".format(model_name))
        
    # Store info of device on which model training has been performed
    if device != None:
        with open('./Models/{}/device.txt'.format(model_name), 'w') as f:
            print(device, file=f)
        
    if hyperparams["loss function"] == F.l1_loss:
        loss = "MAE"
    elif hyperparams["loss function"] == F.mse_loss:
        loss = "MSE"
    elif hyperparams["loss function"] == F.huber_loss:
        loss = "Huber"
    else:
        loss = "None"
        
    N_train = len(train_loader.dataset)
    N_val = len(val_loader.dataset)
    if hyperparams["test set"] == False: 
        N_tot = N_train + N_val
        # Performance Report
        file1 = open("./Models/{}/performance.txt".format(model_name), "w")
        file1.write("Training Process\n")
        file1.write(run_period)
        file1.write("Dataset Size = {}\n".format(N_tot))
        file1.write("Data Split (Train/Val) = {}-{} %\n".format(split_percentage(hyperparams["splits"], hyperparams["test set"])))
        file1.write("Target scaling = {}\n".format(hyperparams["target scaling"]))
        file1.write("Dataset (train+val) mean = {:.6f} eV\n".format(scaling_params[0]))
        file1.write("Dataset (train+val) standard deviation = {:.6f} eV\n".format(scaling_params[1]))
        file1.write("Epochs = {}\n".format(hyperparams["epochs"]))
        file1.write("Batch Size = {}\n".format(hyperparams["batch size"]))
        file1.write("Optimizer = Adam\n")                                            # Kept fixed in this project
        file1.write("Learning Rate scheduler = Reduce Loss On Plateau\n")            # Kept fixed in this project
        file1.write("Initial Learning Rate = {}\n".format(hyperparams["lr0"]))
        file1.write("Minimum Learning Rate = {}\n".format(hyperparams["minlr"]))
        file1.write("Patience (lr-scheduler) = {}\n".format(hyperparams["patience"]))
        file1.write("Factor (lr-scheduler) = {}\n".format(hyperparams["factor"]))
        file1.write("Loss function = {}\n".format(loss))
        file1.close() 
        return None 
    
    N_test = len(test_loader.dataset)  
    N_tot = N_train + N_val + N_test    
    model.eval()
    model.to("cpu")
    
    w_pred = []
    w_true = []
    for batch in test_loader:
        batch = batch.to("cpu")
        w_pred += model(batch)
        w_true += batch.y
    N_test = len(test_loader.dataset)
    train_label_list = []
    for data in train_loader.dataset:
        train_label_list.append(get_graph_formula(data, ENCODER.categories_[0]))
    val_label_list = []
    for data in val_loader.dataset:
        val_label_list.append(get_graph_formula(data, ENCODER.categories_[0]))
    test_label_list  = []    
    for data in test_loader.dataset:
        test_label_list.append(get_graph_formula(data, ENCODER.categories_[0]))
    y_pred = [w_pred[i].item() * std_tv + mean_tv for i in range(N_test)]
    y_true = [w_true[i].item() * std_tv + mean_tv for i in range(N_test)]
    # Histogram based on number of adsorbate atoms (train+val+test dataset)
    n_list = []
    for graph in train_loader.dataset:
        n_list.append(get_number_atoms_from_label(get_graph_formula(graph, ENCODER.categories_[0])))
    for graph in val_loader.dataset:
        n_list.append(get_number_atoms_from_label(get_graph_formula(graph, ENCODER.categories_[0])))
    for graph in test_loader.dataset:
        n_list.append(get_number_atoms_from_label(get_graph_formula(graph, ENCODER.categories_[0])))
    fig, ax = hist_num_atoms(n_list)
    plt.savefig("./Models/{}/num_atoms_hist.svg".format(model_name), bbox_inches='tight')
    plt.close()
    # Violinplot based on chemical family (test set)
    fig, ax = violinplot_family(model, test_loader, std_tv, set(FG_FAMILIES))
    plt.savefig("./Models/{}/test_violin.svg".format(model_name), bbox_inches='tight')
    plt.close()
    # Parity plot (GNN vs DFT) for train, val, test
    my_dict = {"train": train_loader, "val": val_loader, "test": test_loader}
    for key, value in my_dict.items():
        fig, ax = DFTvsGNN_plot(model, value, mean_tv, std_tv)
        plt.savefig("./Models/{}/parity_plot_{}.svg".format(model_name, key), bbox_inches='tight')
        plt.close()
    # Parity plot (GNN vs DFT) for train+val+test together
    fig, ax1, ax2, ax3 = pred_real(model, train_loader, val_loader, test_loader, hyperparams["splits"], mean_tv, std_tv)
    plt.tight_layout()
    plt.savefig("./Models/{}/parity_plot.svg".format(model_name), bbox_inches='tight')
    plt.close()
    # Learning process: MAE vs epoch
    fig, ax = training_plot(train_list, val_list, test_list, hyperparams["splits"])
    plt.savefig("./Models/{}/learning_curve.svg".format(model_name), bbox_inches='tight')
    plt.close()
    # Error analysis
    E = [(y_pred[i] - y_true[i]) for i in range(N_test)]     # Error
    AE = [abs(E[i]) for i in range(N_test)]                  # Absolute Error
    SE = [E[i] ** 2 for i in range(N_test)]                  # Squared Error
    APE = [abs(E[i] / y_true[i]) for i in range(N_test)]     # Absolute Percentage Error
    ME = np.mean(E)                                          # eV
    MAE = np.mean(AE)                                        # eV
    MSE = np.mean(SE)                                        # eV^2
    RMSE = np.sqrt(MSE)                                      # eV
    MAPE = np.mean(APE) * 100.0                              # %
    R2 = r2_score(y_true, y_pred)                            # -
    std_E = np.std(E)                                        # eV
    # Test set: Error distribution plot
    sns.displot(E, bins=50, kde=True)
    plt.tight_layout()
    plt.savefig("./Models/{}/test_error_dist.svg".format(model_name), dpi=DPI, bbox_inches='tight')
    plt.close()
    # Performance Report
    file1 = open("./Models/{}/performance.txt".format(model_name), "w")
    file1.write("Learning Process\n")
    file1.write(run_period)
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("Data Split (Train/Val/Test) = {}-{}-{} %\n".format(*split_percentage(hyperparams["splits"])))
    file1.write("Target scaling = {}\n".format(hyperparams["target scaling"]))
    file1.write("Target (train+val) mean = {:.6f} eV\n".format(mean_tv))
    file1.write("Target (train+val) standard deviation = {:.6f} eV\n".format(std_tv))
    file1.write("Epochs = {}\n".format(hyperparams["epochs"]))
    file1.write("Batch size = {}\n".format(hyperparams["batch size"]))
    file1.write("Optimizer = Adam\n")                                            # Kept fixed in this project
    file1.write("Learning Rate scheduler = Reduce Loss On Plateau\n")            # Kept fixed in this project
    file1.write("Initial learning rate = {}\n".format(hyperparams["lr0"]))
    file1.write("Minimum learning rate = {}\n".format(hyperparams["minlr"]))
    file1.write("Patience (lr-scheduler) = {}\n".format(hyperparams["patience"]))
    file1.write("Factor (lr-scheduler) = {}\n".format(hyperparams["factor"]))
    file1.write("Loss function = {}\n".format(loss))
    file1.write("---------------------------------------------------------\n")
    file1.write("Test Set ({} samples): GNN model performance\n".format(N_test))
    file1.write("Mean Bias Error (MBE) = {:.3f} eV\n".format(ME))
    file1.write("Mean Absolute Error (MAE) = {:.3f} eV\n".format(MAE))
    file1.write("Root Mean Square Error (RMSE) = {:.3f} eV\n".format(RMSE))
    file1.write("Mean Absolute Percentage Error (MAPE) = {:.3f} %\n".format(MAPE))
    file1.write("Error Standard Deviation = {:.3f} eV\n".format(std_E))
    file1.write("R2 = {:.3f} \n".format(R2))
    file1.write("---------------------------------------------------------\n")
    file1.write("Outliers detection (test set, 3*std rule)\n")
    outliers_list = []
    outliers_error_list = []
    index_list = []
    counter = 0
    for sample in range(N_test):
        if abs(E[sample]) >= 3 * std_E:  
            counter += 1
            outliers_list.append(test_label_list[sample])
            outliers_error_list.append(E[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, test_label_list[sample], E[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} eV    (index={})\n".format(counter, test_label_list[sample], E[sample], sample))
            plotter(test_loader.dataset[sample])
            plt.savefig("./Models/{}/Outliers/{}.svg".format(model_name, test_label_list[sample].strip()))
            plt.close()
    file1.close()
    
    # with open("./Models/{}/training_set.csv".format(model_name), "w") as file2:
    #     writer = csv.writer(file2, delimiter='\t')
    #     writer.writerow(["System", "True [eV]", "Prediction [eV]", "Error [eV]"])
    #     writer.writerows(zip(train_label_list, y_true, y_pred, E))
    
    # with open("./Models/{}/validation_set.csv".format(model_name), "w") as file3:
    #     writer = csv.writer(file3, delimiter='\t')
    #     writer.writerow(["System", "True [eV]", "Prediction [eV]", "Error [eV]"])
    #     writer.writerows(zip(val_label_list, y_true, y_pred, E))
            
    with open("./Models/{}/test_set.csv".format(model_name), "w") as file4:
        writer = csv.writer(file4, delimiter='\t')
        writer.writerow(["System", "True [eV]", "Prediction [eV]", "Error [eV]"])
        writer.writerows(zip(test_label_list, y_true, y_pred, E))
    
    return "GNN training saved as {}".format(model_name)
# Hyperparameter optimization directory

In this folder the results of the hyperparameter optimization studies performed with the script `hypopt_GNN.py` are stored. 
Suggestion: Recommended the availability of at least one GPU to run the hyperparamter study as it can take a huge amount of time if performed with the CPU.

For each hyperparameter tuning experiment, a new folder is created here. Each directory contains the following docs:
- `summary.csv`: A .csv file containing a summary of the whole study, each row representing a learning process with a hyperparameter setting picked from the defined hyperparameter space.
- `best_config.csv`: Contains the best hyperparameter setting found based on the defined target to minimize.
- `train_function_*` folders, one for each hyperparameter setting sampled from the hyperparameter space, containing info about the hyperparameters, the stdout, stderr, progress and a .json file with the final result.
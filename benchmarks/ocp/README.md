# Open Catalyst Dataset

## Abstract
This folder aims to provide the needed tools and data to replicate the benchmark
analysis performed by training the models of the Open Catalyst Project (OCP) using the
{FG,BM}-dataset.

## Dependencies
To run the jupyter notebooks inside this folder you will need to install the
dependencies required for GAME-Net plus a working install of the OCP code. To
install it, it is recommended to follow the instructions found in their [Github
repository](https://github.com/Open-Catalyst-Project/ocp).


If your machine has a working install of
[`nix`](https://nixos.org/download.html#nix-install-linux), you can create a
nix-shell that automatically provides the needed dependencies by using the
[`shell.nix`](./shell.nix) file found in this folder through the [`nix-shell`](https://nixos.org/manual/nix/stable/command-ref/nix-shell.html)
command:

``` sh
nix-shell shell.nix
```

Note that [`shell.nix`](./shell.nix) operates with a CPU version of [Geometric
Pytorch](https://pytorch-geometric.readthedocs.io/en/latest/) and it is 
unable to run torch code with a GPU. I strongly recommend to create an alternative
OCP environment with [`conda`](https://docs.conda.io/en/latest/) meeting your
hardware requirements for the training and prediction phases. 

## ./datasets
Datasets directory contains a curated version of both FG and BM datasets
compressed using [`.xz`](https://tukaani.org/xz/xz-file-format-1.1.0.txt) `LZMA`
specification. The internal structure and the naming convention of the datasets
is the following:

```
datasets
├── {group}
│   └── structures
│       ├── {metal}-{name}.contcar
│       └── {metal}-{name}.poscar
├── energies.dat
├── energies_i.dat
└── groups.dat
```

Labeling rules for metal surfaces and gas molecules differ from the ones chosen
for adsorbed molecules: gasses molecules are label as `{name}.(contcar|poscar)`
and surfaces are label as `{metal}-0000.(contcar|poscar)`.
[`lmdb_creation_gamenet.ipynb`](./lmdb_creation_gamenet.ipynb) contains the code
to convert the datasets to the OCP-compatible
[`lmdb`](https://git.openldap.org/openldap/openldap/) format.  If executed, the
notebook will extract the contents of the compressed files, allowing to explore
the raw contents of the databases. Moreover, the notebook contains the procedure
to perform the nested cross validation with an stratified splitting of the
samples.

These datasets only contain the [`POSCAR`](https://www.vasp.at/wiki/index.php/POSCAR) and [`CONTCAR`](https://www.vasp.at/wiki/index.php/CONTCAR) structures (*ansatz* and
relaxed respectively) of each Density Functional Theory (DFT) calculation, the
initial (1 converged SCF cycle) and the relaxed DFT energies, and the chemical
family of each sample. 

## ./predictions
Predictions directory contains a single tarball using the
[`.xz`](https://tukaani.org/xz/xz-file-format-1.1.0.txt) `LZMA` compression. The
tarball encloses the raw
[`.npz`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format)
matrixes with the predictions made by the OCP models pre-trained with the
FG-dataset. These matrixes can be load with the
[`numpy.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
function, and consists in a hashtable-like object with two keys: `ids` and
`energy`, the former indicating the index of the sample and the last the
predicted energy. As the original
[`lmdb`](https://git.openldap.org/openldap/openldap/) are not provided due to
their size, a `ds_data.csv` added for each predictions set. The file contains
the index, name, group and target DFT energy of set separated by commas.

The structure of the predictions folder is the following:

```
predictions
 └── lmdb_(bm|fg)_{geometry}_(poscar|contcar)
    ├── {model}
    │   └── predictions*.npz
    └── ds_data.csv
```

Note that for the BM-dataset predictions only contain a single `predictions.npz`
file, while the FG-dataset ones contain 20 predictions files numbered from 0 to
19 (`predictions_{idx}.csv`).

[`fg_cross_predictions_analysis.ipynb`](./fg_cross_predictions_analysis.ipynb)
and [`bm_predictions_analysis.ipynb`](./bm_predictions_analysis.ipynb) contains
the code to analyze the results obtained for the FG and BM datasets respectively.

## Convert to ensemble
Datasets inside the [`datasets`](./datasets) folder can be converted to ensemble
using the code provided in the [`to_ensemble.ipynb`](./to_ensemble.ipynb) Python
notebook. The code takes the
[`.xz`](https://tukaani.org/xz/xz-file-format-1.1.0.txt) tarball of the datasets
and produces a new tarball with the ensemble structures and the corresponding
ensemble energies.


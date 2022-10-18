# FG-dataset

This directory contains the FG-dataset divided by chemical family. Each directory contains the following files:

1. `structures` folder containing the DFT optimized geometries in VASP CONTCAR format. The file names follow the convention `xx-CHOZ-a.contcar`, where `xx` is the metal symbol, `C`, `H` and `O` are the number of atoms of each element in the adsorbate, `Z` is used progressively from 1 to 9/a,b,c to represent different isomers. the final letter `-a` refers to the adsorption configuration. When the molecules do not contain O or contain multiple heteroatoms, the `O` is replaced by a different symbol. Example: `ag-1401-a` is methane adsorbed on silver, `ru-03N1-a` is ammonia adsorbed on ruthenium. 

2. `energies.dat`: listing the ground state energy got from DFT for all the samples in eV. 

These are the essential data needed at the beginning. All the remaining files are created when the raw data are converted to graphs.
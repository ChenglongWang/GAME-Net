"""Global constants for the GNN project"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Parameters for the geometry conversion to graph
VORONOI_TOLERANCE = 0.25     # Tolerance for the connectivity search
SCALING_FACTOR = 1.2         # Scaling factor for Cu Ir Ni Os Pd Pt Rh Ru 
SCALING_FACTOR_II = 1.5      # Scaling factor for Ag Au Cd Zn
CORDERO_ORIGINAL = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44, 'O' : 0.66,
           'Pd': 1.39, 'P' : 1.07, 'Pt': 1.36, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42, 'Rb': 2.20, 'Ru': 1.46,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22, 'Zr': 1.75}  # DO NOT MODIFY THESE VALUES!
CORDERO = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44*SCALING_FACTOR_II, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32*SCALING_FACTOR, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36*SCALING_FACTOR_II, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41*SCALING_FACTOR, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24*SCALING_FACTOR, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44*SCALING_FACTOR, 'O' : 0.66,
           'Pd': 1.39*SCALING_FACTOR, 'P' : 1.07, 'Pt': 1.36*SCALING_FACTOR, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42*SCALING_FACTOR, 'Rb': 2.20, 'Ru': 1.46*SCALING_FACTOR,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45*SCALING_FACTOR_II,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22*SCALING_FACTOR_II, 'Zr': 1.75}

# Atomic elements considered and one-hot encoder
MOL_ELEM = ['C', 'H', 'O', 'N', 'S']  
METALS = ['Ag', 'Au', 'Cd', 'Cu',  
          'Ir', 'Ni', 'Os', 'Pd',
          'Pt', 'Rh', 'Ru', 'Zn']  
NODE_FEATURES = len(MOL_ELEM) + len(METALS)
ENCODER = OneHotEncoder().fit(np.array(MOL_ELEM + METALS).reshape(-1, 1))  
ELEMENT_LIST = list(ENCODER.categories_[0])                                

# Name of chemical families included in the dataset
FG_RAW_GROUPS = ["amides", "amidines", "group2", "group2b",
                 "group3S", "group3N", "group4", "carbamate_esters",
                 "oximes", "aromatics", "aromatics2",
                 "gas_amides", "gas_amidines", "gas_aromatics",
                 "gas_aromatics2", "gas_carbamate_esters", "gas_group2",
                 "gas_group2b", "gas_group3N", "gas_group3S",
                 "gas_group4", "gas_oximes"]  # Raw Datasets names defined during DFT data generation
FG_FAMILIES = ["Amides", "Amidines", "$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}O_{(0,1)}$",
               "$C_{x}H_{y}S$", "$C_{x}H_{y}N$", "$C_{x}H_{y}O_{(2,3)}$", "Carbamates",
               "Oximes", "Aromatics", "Aromatics", 
               "Amides", "Amidines", "Aromatics", 
               "Aromatics", "Carbamates", "$C_{x}H_{y}O_{(0,1)}$", 
               "$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}N$", "$C_{x}H_{y}S$", 
               "$C_{x}H_{y}O_{(2,3)}$", "Oximes"]  # Proper family used in project (paper, docs, etc)
FAMILY_DICT = dict(zip(FG_RAW_GROUPS, FG_FAMILIES))
OTHER_DATASETS = ["Intermediates", "RPCA"]  # Additional Raw datasets names

# Parameters for plots creation
DPI = 500

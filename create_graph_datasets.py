"""Script to convert DFT samples of the FG dataset to graph formats"""

import argparse

from constants import CORDERO, FG_RAW_GROUPS
from functions import get_tuples, export_tuples, geometry_to_graph_analysis
from paths import paths

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT datasets into graph datasets.")
    PARSER.add_argument("-tol", "--tolerance", type=float, dest="tol", default=0.25,
                        help="Tolerance parameter applied in the graph connectivity search")   
    ARGS = PARSER.parse_args()
    
    TOLERANCE = ARGS.tol
    
    bad_samples = 0
    tot_samples = 0
    print("VORONOI TOLERANCE = {} Angstrom".format(TOLERANCE))
    for dataset in FG_RAW_GROUPS:
        my_tuple = get_tuples(dataset, TOLERANCE, CORDERO)
        export_tuples(paths[dataset]['dataset'], my_tuple)
        x = geometry_to_graph_analysis(dataset)
        if dataset[:3] != "gas":
            bad_samples += x[0]
            tot_samples += x[2]
    print("Voronoi tolerance: {} Angstrom".format(TOLERANCE))
    print("Bad samples: {}".format(bad_samples))
    print("Total samples: {}".format(tot_samples))
    print("Percentage of bad samples: {:.2f}%".format(bad_samples * 100/tot_samples))
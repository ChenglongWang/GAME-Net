"""Script to convert DFT samples of the FG dataset to graph formats"""

import argparse
from paths import paths

from constants import FG_RAW_GROUPS, GRAPH_REP_PARAMS
from functions import get_tuples, export_tuples, geometry_to_graph_analysis

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT datasets into graph datasets using the parameters provided in constants.py \
        .The output of this script is the creation of a Dataset.dat file for each chemical family, with the graph information.")
        
    bad_samples = 0
    tot_samples = 0
    print("Conversion algorithm DFT -> Graph")
    print("Voronoi tolerance = {} Angstrom".format(GRAPH_REP_PARAMS["Voronoi_tolerance"]))
    print("2nd order metal neighbours inclusion = {}".format(GRAPH_REP_PARAMS["Second_order_nn"]))
    print("Scaling factor I = {}".format(GRAPH_REP_PARAMS["Scaling_factor_I"]))
    print("Scaling factor II = {}".format(GRAPH_REP_PARAMS["Scaling_factor_II"]))
    
    for dataset in FG_RAW_GROUPS:
        tuple = get_tuples(dataset,
                           GRAPH_REP_PARAMS["Voronoi_tolerance"],
                           GRAPH_REP_PARAMS["Atomic_radii"], 
                           GRAPH_REP_PARAMS["Second_order_nn"])
        export_tuples(paths[dataset]['dataset'], tuple)
        x = geometry_to_graph_analysis(dataset)
        if dataset[:3] != "gas":
            bad_samples += x[0]
            tot_samples += x[2]
        else:
            bad_samples += 0  # Gas-phase molecules are always correctly converted
            tot_samples += x
    print("Bad conversions: {}".format(bad_samples))
    print("Total samples: {}".format(tot_samples))
    print("Percentage of bad conversions: {:.2f}%".format(bad_samples * 100/tot_samples))
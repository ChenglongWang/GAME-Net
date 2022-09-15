"""Script to convert DFT samples of the FG dataset to graph formats"""

import argparse
from paths import paths
import toml

from constants import FG_RAW_GROUPS
from functions import get_tuples, export_tuples, geometry_to_graph_analysis

def create_graph_datasets(voronoi_tolerance: float,
                          second_order_nn: bool, 
                          scaling_factor: float):
    """Convert raw DFT data into graph representation that will be saved in a txt file
    Args:
        voronoi_tolerance (float): Tolerance applied in the tessellation algorithm for edge definition
        second_order_nn (bool): Whether to include the 2-hop metal atoms neighbours 
        scaling_factor (float): Scaling factor applied to metals
    Returns:
        _type_: _description_
    """
    bad_samples = 0
    tot_samples = 0
    print("GEOMETRY -> GRAPH CONVERSION")
    print("Voronoi tolerance = {} Angstrom".format(voronoi_tolerance))
    print("2nd order metal neighbours inclusion = {}".format(second_order_nn))
    print("Scaling factor for metal atomic radius= {}".format(scaling_factor))
    for dataset in FG_RAW_GROUPS:
        tuple = get_tuples(dataset, voronoi_tolerance, second_order_nn, scaling_factor)
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
    return bad_samples, tot_samples
    
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert raw DFT datasets into graph representations with the parameters provided in config.toml.")
    params = toml.load("config.toml")["graph_params"]
    create_graph_datasets(params["voronoi_tol"], params["second_order_nn"], params["scaling_factor"])
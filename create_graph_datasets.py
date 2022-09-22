"""Script to convert DFT samples of the FG dataset to graph formats"""

import argparse
import toml

from functions import get_tuples, export_tuples, geometry_to_graph_analysis

def create_graph_datasets(graph_settings: dict, 
                          paths_dict: dict):
    """Convert raw DFT data into pre-processed graph datasets.
    Args:
        graph_settings (dict): {"voronoi_tol":0.5, "scaling_factor":1.5, "second_order_nn": False}
            voronoi_tolerance (float): Tolerance of the tessellation algorithm for edge creation
            second_order_nn (bool): Whether to include the 2-hop metal atoms neighbours 
            scaling_factor (float): Scaling factor applied to metal atomic radii
        paths_dict (dict): Data paths
    Returns:
        _type_: _description_
    """
    bad_samples = 0
    tot_samples = 0
    voronoi_tolerance = graph_settings["voronoi_tol"]
    second_order_nn = graph_settings["second_order_nn"]
    scaling_factor = graph_settings["scaling_factor"]
    print("GEOMETRY -> GRAPH CONVERSION")
    print("Voronoi tolerance = {} Angstrom".format(voronoi_tolerance))
    print("2-hop metal neighbours = {}".format(second_order_nn))
    print("Scaling factor = {}".format(scaling_factor))
    chemical_families = list(paths_dict.keys())
    chemical_families.remove('metal_surfaces')
    for chemical_family in chemical_families:
        tuple = get_tuples(chemical_family, 
                           voronoi_tolerance,
                           second_order_nn,
                           scaling_factor,
                           paths_dict)
        export_tuples(paths_dict[chemical_family]['dataset'],
                      tuple)  # Generate pre_xx_bool_xxx.dat 
        x = geometry_to_graph_analysis(chemical_family, paths_dict)
        bad_samples += x[0]
        tot_samples += x[2]
    print("Bad conversions: {}/{} ({:.2f}%)".format(bad_samples, tot_samples, bad_samples*100/tot_samples))
    return bad_samples, tot_samples
    
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert raw DFT datasets into graph representations with the parameters provided in config.toml.")
    params = toml.load("config.toml")["graph_params"]
    create_graph_datasets(params["voronoi_tol"], params["second_order_nn"], params["scaling_factor"])
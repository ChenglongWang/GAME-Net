import argparse
import sys
sys.path.insert(0, "../src")

from gnn_eads.functions import get_id
from gnn_eads.paths import create_paths
from gnn_eads.constants import FG_RAW_GROUPS
from gnn_eads.create_graph_datasets import create_graph_datasets

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Convert DFT samples of the FG dataset to graph formats')
    PARSER.add_argument('--data', type=str, dest="data", help='Path to data folder')
    PARSER.add_argument('--tol', type=float, default=0.5, dest="tol", help='Voronoi tolerance')
    PARSER.add_argument('--second_order_nn', type=bool, default=False, dest="sec_nn", help='Whether to include the 2-hop metal atoms neighbours')
    PARSER.add_argument('--scaling_factor', type=float, default=1.5, dest="sf", help='Scaling factor applied to metal atomic radii')
    ARGS = PARSER.parse_args()
    graph_settings = {"voronoi_tol":ARGS.tol, "scaling_factor":ARGS.sf, "second_order_nn":ARGS.sec_nn}
    id = get_id(graph_settings)
    paths = create_paths(FG_RAW_GROUPS, ARGS.data, id)
    create_graph_datasets(graph_settings, paths)
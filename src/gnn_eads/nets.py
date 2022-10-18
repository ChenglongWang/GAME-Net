"""Module containing the Graph Neural Network architectures."""

import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GraphMultisetTransformer
from torch_geometric.data import Data

from gnn_eads.constants import NODE_FEATURES
from gnn_eads.functions import get_graph_conversion_params, get_mean_std_from_model

class Net_TEMPLATE(torch.nn.Module):
    """Template for creating specific GNN architecture.
    Args:
        dim(int): depth of hidden layers. Default to 128.
        node_features(int): number of features per node. Default to 17.
        sigma(torch.nn): activation function used in the GNN. Default is the ReLU.
        bias(bool): switch for turning on the additive bias in the GNN layers. Default to False.    
    """
    
    def __init__(self, dim=128, node_features=17, sigma=torch.nn.ReLU(), bias=False):
        super(Net_TEMPLATE, self).__init__()
        self.dim = dim
        self.node_features = node_features
        self.sigma = sigma
        self.bias = bias  
        #------------------------------------------------
        # Instantiation of the building blocks of the GNN
        #------------------------------------------------

    def forward(self, data):
        #-----------------------------
        # NODE LEVEL (MESSAGE-PASSING)
        #-----------------------------
        
        #----------------------
        # GRAPH LEVEL (POOLING)
        #----------------------
        pass
        
class SantyxNet(torch.nn.Module):
    """
    Architecture found before hyperpaeameter optimization.
    Args:
        dim(int): depth of hidden layers. Default to 128.
        node_features(int): number of features per node. Default to 17.
    """    
    def __init__(self, dim: int=128,
                 node_features: int=NODE_FEATURES,
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv_normalize: bool=False,
                 conv_root_weight: bool=True, 
                 pool_ratio: float=0.25, 
                 pool_heads: int=2, 
                 pool_seq: list[str]=["GMPool_G", "SelfAtt", "GMPool_I"], 
                 pool_layer_norm: bool=False):
        super(SantyxNet, self).__init__()
        self.dim = dim
        self.node_features = node_features
        self.sigma = sigma
        #------------------------------------------------
        # Instantiation of the building blocks of the GNN
        #------------------------------------------------
        self.lin1 = Linear(self.node_features, self.dim, bias=bias)
        self.lin2 = Linear(self.dim, self.dim, bias=bias)
        self.lin3 = Linear(self.dim, self.dim, bias=bias)
        self.lin4 = Linear(self.dim, self.dim, bias=bias)
        self.lin5 = Linear(self.dim, self.dim, bias=bias)
        self.lin6 = Linear(self.dim, self.dim, bias=bias)
        self.conv1 = SAGEConv(self.dim, self.dim, bias=bias, normalize=conv_normalize, root_weight=conv_root_weight)
        self.conv2 = SAGEConv(self.dim, self.dim, bias=bias, normalize=conv_normalize, root_weight=conv_root_weight)
        self.conv3 = SAGEConv(self.dim, self.dim, bias=bias, normalize=conv_normalize, root_weight=conv_root_weight)
        self.pool = GraphMultisetTransformer(self.dim,              # self.dim
                                             self.dim,              # self.dim
                                             1,                     # 1 
                                             num_nodes=100,         # 300
                                             pooling_ratio=pool_ratio,
                                             pool_sequences=pool_seq,
                                             num_heads=pool_heads,           
                                             layer_norm=pool_layer_norm)                                                                                    
        
    def forward(self, data):
        #-----------------------------
        # NODE LEVEL (MESSAGE-PASSING)
        #-----------------------------        
        out = self.sigma(self.lin1(data.x))   
        out = self.sigma(self.lin2(out))
        out = self.sigma(self.lin3(out))
        out = self.sigma(self.conv1(out, data.edge_index))
        out = self.sigma(self.lin4(out))
        out = self.sigma(self.conv2(out, data.edge_index))
        out = self.sigma(self.lin5(out))
        out = self.sigma(self.conv3(out, data.edge_index))
        out = self.sigma(self.lin6(out))
        #----------------------
        # GRAPH LEVEL (POOLING)
        #----------------------
        out = self.pool(out, data.batch, data.edge_index)
        return out.view(-1)
    
class FlexibleNet(torch.nn.Module):
    def __init__(self, 
                 dim: int=128,                  
                 N_linear: int=3,
                 N_conv: int=3,
                 adj_conv: bool=True,
                 in_features: int=NODE_FEATURES,                 
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv,
                 pool=GraphMultisetTransformer, 
                 pool_ratio: float=0.25, 
                 pool_heads: int=4, 
                 pool_seq: list[str]=["GMPool_G", "SelfAtt", "GMPool_I"], 
                 pool_layer_norm: bool=False):
        """Flexible Net for Hyperparamater optimization

        Args:
            dim (int, optional): Layer depth. Defaults to 128.
            N_linear (int, optional): Number of NN layers. Defaults to 3.
            N_conv (int, optional): Number of convolutional layers. Defaults to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Defaults to True.
            in_features (int, optional): Input graph node dimensionality. Defaults to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Defaults to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Defaults to True.
            conv (_type_, optional): Convolutional Layer. Defaults to SAGEConv.
            pool (_type_, optional): Pooling Layer. Defaults to GraphMultisetTransformer.
        """
        super(FlexibleNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Instantiation of the building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.inlin1 = Linear(self.dim, self.dim, bias=bias)
        self.inlin2 = Linear(self.dim, self.dim, bias=bias)
        self.inlin3 = Linear(self.dim, self.dim, bias=bias)
        self.inlin4 = Linear(self.dim, self.dim, bias=bias)
        self.inlin5 = Linear(self.dim, self.dim, bias=bias)
        self.inlin6 = Linear(self.dim, self.dim, bias=bias)
        self.inlin7 = Linear(self.dim, self.dim, bias=bias)
        self.lin_block = [self.inlin1, self.inlin2, self.inlin3, self.inlin4, self.inlin5, self.inlin6, self.inlin7]
        
        self.lin1 = Linear(self.dim, self.dim, bias=bias)
        self.lin2 = Linear(self.dim, self.dim, bias=bias)
        self.lin3 = Linear(self.dim, self.dim, bias=bias)
        self.lin4 = Linear(self.dim, self.dim, bias=bias)
        self.lin5 = Linear(self.dim, self.dim, bias=bias)
        self.lin6 = Linear(self.dim, self.dim, bias=bias)
        self.conv1 = conv(self.dim, self.dim, bias=bias)
        self.conv2 = conv(self.dim, self.dim, bias=bias)
        self.conv3 = conv(self.dim, self.dim, bias=bias)
        self.conv4 = conv(self.dim, self.dim, bias=bias)
        self.conv5 = conv(self.dim, self.dim, bias=bias)
        self.conv6 = conv(self.dim, self.dim, bias=bias)
        self.adj_block = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, self.lin6]
        self.conv_block = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        self.pool = pool(self.dim, self.dim, 1, 
                         num_nodes=300,         # 300
                         pooling_ratio=pool_ratio,
                         pool_sequences=pool_seq,
                         num_heads=pool_heads,           
                         layer_norm=pool_layer_norm)                                                             
        
    def forward(self, data):
        #-----------------------------
        # NODE LEVEL (MESSAGE-PASSING)
        #-----------------------------        
        out = self.sigma(self.input_layer(data.x))   
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.lin_block[layer](out))
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.lin_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #----------------------
        # GRAPH LEVEL (POOLING)
        #----------------------
        out = self.pool(out, data.batch, data.edge_index)
        return out.view(-1)

class PreTrainedModel():
    def __init__(self, model_path: str):
        """Container class for loading pre-trained GNN models on the cpu.
        Args:
            model_path (str): path to model folder.
        """
        self.model_path = model_path
        self.model = torch.load("{}/model.pth".format(self.model_path),
                                map_location=torch.device("cpu"))
        self.model.load_state_dict(torch.load("{}/GNN.pth".format(self.model_path), 
                                              map_location=torch.device("cpu")))
        self.model.eval()  # Inference mode
        self.model.to("cpu")
        # Scaling parameters
        self.mean, self.std = get_mean_std_from_model(self.model_path)
        # Graph conversion parameters
        self.g_tol, self.g_sf, self.g_metal_2nn = get_graph_conversion_params(self.model_path)
    
    def evaluate(self, graph: Data) -> float:
        """Evaluate graph energy

        Args:
            graph (Data): adsorption/molecular graph

        Returns:
            float: system energy in eV
        """
        return self.model(graph).item() * self.std + self.mean
    
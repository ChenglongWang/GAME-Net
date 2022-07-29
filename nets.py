"""Module containing the Graph Neural Network architectures."""

__author__ = "Santiago Morandi"

import torch
from torch.nn import Linear, GRU 
from torch_geometric.nn import Set2Set, CGConv, Set2Set, SAGEConv, GraphMultisetTransformer
import torch.nn.functional as F

from constants import NODE_FEATURES
class Net_TEMPLATE(torch.nn.Module):
    """Reference GNN class template
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
      
class Net(torch.nn.Module):
    def __init__(self, dim=128, node_features=17):
        """
        The layers of the GNN are defined (order here doesn't matter!)
        Original model proposed by Sergio. DO NOT TOUCH IT
        Args:
            dim(int): depth of the GNN layers. Default to 128.
            node_features(int): original number of node features. Default to 17.
        """
        super(Net, self).__init__()
        self.dim = dim
        self.node_features = node_features
        # GNN layers definition
        self.lin0 = Linear(self.node_features, self.dim)
        self.conv1 = CGConv(self.node_features, 0, aggr='add')
        self.conv2 = CGConv(self.dim, 0, aggr='add')
        self.conv3 = CGConv(self.dim, 0, aggr='max')
        self.gru1 = GRU(self.dim, self.dim, num_layers=1)
        self.gru2 = GRU(self.dim, self.dim, num_layers=1)
        self.set2set = Set2Set(self.dim, processing_steps=3)
        self.lin1 = Linear(self.dim, self.dim)
        self.lin2 = Linear(2 * self.dim, self.node_features)
        self.lin3 = Linear(self.node_features, 1)
        self.connect = Linear(self.dim * 2, self.dim)

    def forward(self, data):
        """
        Here the structure of the GNN is defined (order matters!)
        """
        sigma = F.relu
        out = sigma(self.conv1(data.x, data.edge_index))
        out = sigma(self.lin0(out))
        out = sigma(self.lin1(out))
        out2 = out
        h = out.unsqueeze(0)
        j = h
        for i in range(3):
            m = self.conv2(out, data.edge_index)
            n = self.conv3(out2, data.edge_index)
            out, h = self.gru1(m.unsqueeze(0), h)
            out2, j = self.gru2(n.unsqueeze(0), j)
            out = out.squeeze(0)
            out2 = out.squeeze(0)
        comn = torch.cat((out, out2), -1)
        out = sigma(self.connect(comn))
        out = self.set2set(out, data.batch)
        out = sigma(self.lin2(out)) 
        out = self.lin3(out) 
        return out.view(-1)  
        
class SantyxNet(torch.nn.Module):
    """
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
                 pool_heads: int=4, 
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
                                             num_nodes=300,         # 300
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

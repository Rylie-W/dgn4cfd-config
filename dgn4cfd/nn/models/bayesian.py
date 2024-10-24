import torch
from torch import nn
import bayesian_torch.layers as bnn_layers

from .vanilla import VanillaGnn
from ..blocks import FNN
from ...graph import Graph


class BayesianGnn(VanillaGnn):
    """ Bayesian GNN model.

    Args:
        arch (dict): Dictionary with the architecture of the model. It must contain the following keys:
            - 'in_node_features' (int): Number of input node features. This is the number of features of the noisy field.
            - 'cond_node_features' (int, optional): Number of conditional node features. Defaults to 0.
            - 'cond_edge_features' (int, optional): Number of conditional edge features. Defaults to 0.
            - 'in_edge_features' (int, optional): Number of input edge features. Defaults to 0.
            - 'depths' (list): List of integers with the number of layers at each depth.
            - 'fnns_depth' (int, optional): Number of layers in the FNNs. Defaults to 2.
            - 'fnns_width' (int): Width of the FNNs.
            - 'aggr' (str, optional): Aggregation method. Defaults to 'mean'.
            - 'dropout' (float, optional): Dropout probability. Defaults to 0.0.
            - 'activation' (torch.nn.Module, optional): Activation function. Defaults to torch.nn.SELU.
            - 'pooling_method' (str, optional): Pooling method. Defaults to 'interp'.
            - 'unpooling_method' (str, optional): Unpooling method. Defaults to 'uniform'.
            - 'dim' (int, optional): Dimension of the latent space. Defaults to 2.
            - 'scalar_rel_pos' (bool, optional): Whether to use scalar relative positions. Defaults to True.    
    """

    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        super().load_arch(arch)
        # Transform the Linear layers in the encoder and decoder into BayesianLinear
        self.edge_encoder = self.BayesianLinear(self.edge_encoder)
        self.node_encoder = self.BayesianLinear(self.node_encoder)
        self.node_decoder = self.BayesianLinear(self.node_decoder)
        # Transform the Linear layers in the FNNs into BayesianLinear layers
        for module in self.modules():
            if isinstance(module, FNN):
                layers = module.layers
                for i, layer in enumerate(layers):
                    if isinstance(layer, nn.Linear):
                        module.layers[i] = self.BayesianLinear(module.layers[i])

    class BayesianLinear(bnn_layers.LinearReparameterization):
        def __init__(
            self,
            linear: nn.Linear,
        ) -> None:
            super().__init__(linear.in_features, linear.out_features)
            self.to(linear.weight.data.device)

        def forward(self, x):
            return super().forward(x, return_kl=False)
        
    def kl_loss(self):
        return torch.mean(torch.stack([module.kl_loss() for module in self.modules() if isinstance(module, bnn_layers.LinearReparameterization)]))
    
    @torch.no_grad()
    def sample(
        self, 
        graph: Graph,
        seed:  int = None,
    ) -> torch.Tensor:
        self.eval()
        graph = graph.to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        return self(graph)
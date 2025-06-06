import torch
from torch import nn

from ..flow_matching_model import FlowMatchingModel
from ...blocks import SinusoidalTimeEmbedding
from ...models.multi_scale_gnn import MultiScaleGnn
from ...models.vgae import VGAE
from ....graph import Graph


class LatentFlowMatchingGraphNet(FlowMatchingModel):
    r"""Defines a GNN that approximates the advection vector field of a flow matching model in the latent space of a VGAE.

    Args:
        autoencoder_checkpoint (str): Path to the checkpoint of the VGAE.
        arch (dict): Dictionary with the architecture of the model. It must contain the following keys:
            - 'in_node_features' (int): Number of input node features (latent node features from the VGAE's encoder).
            - 'cond_node_features' (int): Number of conditional node features (latent conditional node features from the VGAE's condition encoder).
            - 'cond_edge_features' (int): Number of conditional edge features (latent conditional edge features from the VGAE's condition encoder).
            - 'depths' (list): List of integers with the number of layers at each depth.
            - 'fnns_depth' (int, optional): Number of layers in the FNNs. Defaults to 2.
            - 'fnns_width' (int): Width of the FNNs.
            - 'activation' (torch.nn.Module, optional): Activation function. Defaults to nn.SELU.
            - 'aggr' (str, optional): Aggregation method. Defaults to 'mean'.
            - 'dropout' (float, optional): Dropout probability. Defaults to 0.0.
            - 'emb_width' (int, optional): Width of the diffusion-step embedding. Defaults to 4 * fnns_width.
            - 'dim' (int, optional): Number of spatial dimensions of the physical space. Defaults to 2.
            - 'scalar_rel_pos' (bool, optional): Whether to use scalar relative positions between nodes in HR and LR graphs. Defaults to True.
    """

    def __init__(
        self,
        autoencoder_checkpoint: str,
        *args,
        **kwargs
    ) -> None: 
        self.autoencoder_checkpoint = autoencoder_checkpoint
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Hyperparameters
        self.in_node_features   = arch['in_node_features']
        self.cond_node_features = arch.get('cond_node_features')
        self.cond_edge_features = arch.get('cond_edge_features')
        self.depths             = arch['depths']
        self.fnns_depth         = arch.get('fnns_depth', 2)
        self.fnns_width         = arch['fnns_width']
        self.aggr               = arch.get('aggr', 'mean')
        self.dropout            = arch.get('dropout', 0.0)
        self.emb_width          = arch.get('emb_width', self.fnns_width * 4)
        self.dim                = arch.get('dim', 2)
        self.scalar_rel_pos     = arch.get('scalar_rel_pos', True)
        # Validate the inputs
        assert self.in_node_features > 0, "Input node features must be a positive integer"
        assert self.cond_node_features >= 0, "Condition features must be a non-negative integer"
        assert len(self.depths) > 0, "Depths (`depths`) must be a list of integers"
        assert isinstance(self.depths, list), "Depths (`depths`) must be a list of integers"
        assert all([isinstance(depth, int) for depth in self.depths]), "Depths (`depths`) must be a list of integers"
        assert all([depth > 0 for depth in self.depths]), "Depths (`depths`) must be a list of positive integers"
        assert self.fnns_depth >=2 , "FNNs depth (`fnns_depth`) must be at least 2"
        assert self.fnns_width > 0, "FNNs width (`fnns_width`) must be a positive integer"
        assert self.aggr in ('mean', 'sum'), "Aggregation method (`aggr`) must be either 'mean' or 'sum'"
        assert self.dropout >= 0.0 and self.dropout < 1.0, "Dropout (`dropout`) must be a float between 0.0 and 1.0"
        self.out_node_features = self.in_node_features
        # Load and freeze the autoencoder
        self.autoencoder = VGAE(
            checkpoint = self.autoencoder_checkpoint,
            device     = self.device,
        )
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        # Initial scale of the DiffusionGnn
        self.scale_0 = len(self.autoencoder.arch['depths']) - 1
        # r embedding
        self.r_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.fnns_width),
            nn.Linear(self.fnns_width, self.emb_width),
            nn.SELU(),
        )
        # Node encoder
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features,
            out_features = self.fnns_width,
        )
        self.cond_encoder = nn.Linear(
            in_features  = self.cond_node_features,
            out_features = self.fnns_width,
        )
        
        # r encoder
        self.r_encoder = nn.ModuleList([
            nn.Linear(self.emb_width, self.fnns_width),      # Applied to the r embedding
            nn.SELU(),                                       # Applied to the previous output and the node encoder output
            nn.Linear(self.fnns_width * 2, self.fnns_width), # Applied to the previous output
        ])
        # Edge encoder
        self.edge_encoder = nn.Linear(
            in_features  = self.cond_edge_features,
            out_features = self.fnns_width,
        )
        # MuS-GNN propagator
        self.propagator = MultiScaleGnn(
            depths            = self.depths,
            fnns_depth        = self.fnns_depth,
            fnns_width        = self.fnns_width,
            emb_features      = self.emb_width,
            aggr              = self.aggr,
            activation        = nn.SELU,
            dropout           = self.dropout,
            scale_0           = self.scale_0,
            dim               = self.dim,
            scalar_rel_pos    = self.scalar_rel_pos,
        )
        # Node decoder
        self.node_decoder = nn.Linear(
            in_features  = self.fnns_width,
            out_features = self.out_node_features,
        )
 
    @property
    def num_fields(self) -> int:
        return self.out_node_features
    
    def reset_parameters(self):
            modules = [module for module in self.children() if hasattr(module, 'reset_parameters')]
            for module in modules:
                module.reset_parameters()

    def forward(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        assert hasattr(graph, 'c_latent'), "graph must have an attribute 'cond' indicating the condition"
        assert hasattr(graph, 'e_latent'), "The latent edge features (`e_latent`) must be provided."
        assert hasattr(graph, 'r'), "graph must have an attribute 'r'"
        assert hasattr(graph, 'field_r'), "graph must have an attribute 'field_r'"
        # Embed r
        emb = self.r_embedding(graph.r) # Shape (batch_size, emb_width)
        # Encode the node features
        x_latent = self.node_encoder(graph.field_r) + self.cond_encoder(graph.c_latent)
        # Encode the diffusion step embedding into the node features
        emb_proj = self.r_encoder[0](emb)
        x_latent = torch.cat([x_latent, emb_proj[graph.batch]], dim=1)
        for layer in self.r_encoder[1:]:
            x_latent = layer(x_latent)
        # Encode the edge features
        e_latent = self.edge_encoder(graph.e_latent)
        # Propagate the latent space
        x_latent, _ = self.propagator(graph, x_latent, e_latent, emb)
        # Decode the latent node features
        return self.node_decoder(x_latent)
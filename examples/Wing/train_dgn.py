"""
    Train a Diffusion Graph Network (DGN) to predict the pressure field on a wing.
    Run with:
        python train_dgn.py --experiment_id 0 --gpu 0
"""


import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn

torch.multiprocessing.set_sharing_strategy('file_system')


argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--gpu',           type=int, default=0)
args = argparser.parse_args()

# Initial seed
seed = 0
torch.manual_seed(seed)

# Dictionary of experiments
experiment = {
    0: {
        'name': 'MODEL_NAME',
        'nt':    250, # Limit the length of the training simulations to 250 time-steps
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints',
    # checkpoint    = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.HybridLoss(),
    epochs        = 5000,
    batch_size    = 32,
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 250, "loss": 'training'},
    stopping      = 1e-8,
    step_sampler  = dgn.nn.diffusion.ImportanceStepSampler,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
transform = transforms.Compose([
    dgn.transforms.ScaleEdgeAttr(0.015),                        # Scale the relative position stored as `edge_attr`
    dgn.transforms.EdgeCondFreeStream(normals='loc'),           # Add the projection of the free stream velocity along edge-local axes as `edge_cond`
    dgn.transforms.ScaleAttr('target', vmin=-1850,  vmax=400),  # Scale the target field (pressure)
    dgn.transforms.MeshCoarsening(                              # Create 5 lower-resolution graphs and normalise the relative position betwen the inter-graph nodes.
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True, 
    ),
])
dataset = dgn.datasets.pOnWing(
    path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnWingTrain).file_path,
    T         = experiment['nt'],
    transform = transform,
    preload   = False,
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 16,    
)   

# Diffusion process
diffusion_process = dgn.nn.diffusion.DiffusionProcess(
    num_steps     = 1000,
    schedule_type = 'linear',
)

# Model
arch = {
    'dim':                3, # 3D
    'in_node_features':   1, # Noisy p
    'cond_node_features': 3, # nx, ny, nz
    'cond_edge_features': 6, # x_j - x_i and U_\inf on local edge axes
    'depths':             6 * [2,],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
}
model = dgn.nn.DiffusionGraphNet(
    diffusion_process  = diffusion_process,
    learnable_variance = True,
    arch               = arch
)

# Training
model.fit(train_settings, dataloader)
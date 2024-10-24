"""
    Run with:
        python train_ldgn.py --experiment_id 0 --gpu 0
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
        'name':        'MODEL_NAME',
        'autoencoder': './checkpoints/AE_NAME.chk',
        'depths':      [1,2,2,2],
        'nt':          250,
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints',
    # checkpoint    = './checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.HybridLoss(),
    epochs        = 50000,
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
    dgn.transforms.ScaleEdgeAttr(0.015),
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    dgn.transforms.ScaleAttr('target', vmin=-1850,  vmax=400),
    dgn.transforms.MeshCoarsening(
        num_scales      = 6,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4],
        scalar_rel_pos  = True, 
    ),
])
dataset = dgn.datasets.pOnWing(
    path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnWingTrain).file_path,
    T         = experiment['nt'],
    transform = transform,
    preload   = True,
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
    'dim':                3,   # 3D
    'in_node_features':   1,   # Noisy latent features
    'cond_node_features': 126, # Latent node features with conditional information
    'cond_edge_features': 126, # Latent edge features with conditional information
    'depths':             experiment['depths'],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
}
model = dgn.nn.LatentDiffusionGraphNet(
    autoencoder_checkpoint = experiment['autoencoder'],
    diffusion_process      = diffusion_process,
    learnable_variance     = True,
    arch                   = arch
)    

# Training
model.fit(train_settings, dataloader)
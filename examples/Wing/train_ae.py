"""
    Run with:
        python train_ae.py --experiment_id 0 --gpu 0
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
        'name':                 'MODEL_NAME',
        'latent_node_features': 1,
        'kl_reg':               1e-6,
        'fnns_width':           126,
        'depths':               [2,2,1],
        'nt':                    250,
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints',
    # checkpoint    = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.VaeLoss(kl_reg=experiment['kl_reg']),
    epochs        = 5000,
    batch_size    = 32,
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 250, "loss": 'training'},
    stopping      = 1e-8,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
transform = transforms.Compose([
    dgn.transforms.ScaleEdgeAttr(0.015),
    dgn.transforms.EdgeCondFreeStream(normals='loc'),
    dgn.transforms.ScaleAttr('target', vmin=-1850,  vmax=400),
    dgn.transforms.MeshCoarsening(
        num_scales      = 4,
        rel_pos_scaling = [0.015, 0.03, 0.06, 0.12],
        scalar_rel_pos  = True, 
    ),
    dgn.transforms.Copy('target', 'field'), # Because the target is the input field
])
dataset = dgn.datasets.pOnWing(
    path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnWingTrain).file_path,
    T         = experiment['nt'],
    transform = transform,
    preload   = True
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 16,
)   

# Model
arch = {
    'dim':                  3, # 3D
    'in_node_features':     1, # p
    'cond_node_features':   3, # nx, ny, nz
    'cond_edge_features':   6, # x_j - x_i, U_\inf on local edge axes
    'latent_node_features': experiment['latent_node_features'],
    'depths':               experiment['depths'],
    'fnns_width':           experiment['fnns_width'],
    'aggr':                 'sum',
    'dropout':              0.1,
    'norm_latents':         True,
}
model = dgn.nn.VGAE(arch = arch)

# Training
model.fit(train_settings, dataloader)
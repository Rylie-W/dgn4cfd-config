# DGN4CFD: Diffusion Graph Nets for Computational Fluid Dynamics

<p align="center">
  <img src="https://i.ibb.co/G2JqcCN/DGN-Ellipse-Flow-compressed.gif"  width="800" />
</p>
<p align="center">
  <em>Figure 1: Velocity and pressure fields around an elliptical cylinder sampled from a DGN</em>
</p>
<br>

<p align="center">
  <img src="https://i.ibb.co/DpNPLmm/dgn-wing.gif" width="400" />
</p>
<p align="center">
  <em>Figure 2: Pressure field on a wing model sampled from a DGN</em>
</p>
<br>

## Installation
Python 3.10 or higher and [PyTorch](https://pytorch.org/) 2.4 or higher are required.
We recommend installing **dgn4cfd** in a virtual environment, e.g., using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
It can installed from the current directory by running:

```bash
pip install -e .
```

This also installs [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and compiles [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster), so it may take a while.
Once **dgn4cfd** has been installed, it can be imported in Python as follows:

```python
import dgn4cfd as dgn
```

## Datasets
Datasets can be downloaded directly within python using our `DatasetDownloader` and then we can create the dataset object by indicating the download path:
```python
import dgn4cfd as dgn

downloader = dgn.datasets.DatasetDownloader(
    dataset_url = dgn.datasets.DatasetUrl.<DATASET NAME>,
    path        = <DOWNLOAD PATH>

)

dataset = dgn.datasets.pOnEllipse(
    path       = downloader.file_path,
    T          = <LENGTH OF SIMULATIONS>,
    transform  = <PRE-PROCESSING TRANSFORMATIONS>,
)

graph = dataset[<SIMULATION IDX>]
```

The datasets (`<DATASET NAME>`) available (so far) are:

- **pOnEllipse task**: Infer pressure on the surface of an ellipse immersed on a laminar flow ($Re \in [500, 1000]$ in the training dataset). Each simulation has 101 time-steps (`<LENGTH OF SIMULATIONS>`= 101). The training and testing datasets are:
  - pOnEllipseTrain (Training dataset)
  - pOnEllipseInDist
  - pOnEllipseLowRe
  - pOnEllipseHighRe
  - pOnEllipseThin
  - pOnEllipseThick

- **uvpAroundEllipse task**: Infer the velocity and pressure fields around an ellipse immersed on a laminar flow ($Re \in [500, 1000]$ in the training dataset).  Each simulation has 101 time-steps (`<LENGTH OF SIMULATIONS>`= 101). The training and testing datasets are:
  - uvpAroundEllipseTrain (Training dataset, 30.1 GB)
  - uvpAroundEllipseInDist
  - uvpAroundEllipseLowRe
  - uvpAroundEllipseHighRe
  - uvpAroundEllipseThin
  - uvpAroundEllipseThick

- **pOnWing task**: Infer pressure on the surface of a wing in 3D turbulent flow. The wing cross section is NACA 24XX airfoil. The geometry of the wings varies in terms of relative thickness, taper ratio, sweep angle, and twist angle. The training simulations have 251 time-steps each (`<LENGTH OF SIMULATIONS>`= 251) and the test simulations have 2501 time-steps each (`<LENGTH OF SIMULATIONS>`= 2501). The training and testing datasets are:
  - pOnWingTrain (Training dataset, 6.52 GB)
  - pOnWingInDist
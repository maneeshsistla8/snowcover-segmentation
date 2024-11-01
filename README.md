# snowcover-segmentation
A pipeline for training and evaluating a UNet model for snow cover segmentation using satellite imagery.

This was developed as part of the machine learning research efforts to estimate the dynamic snow cover in the Salt-Verde region in Arizona, undertaken by the Center of Hydrologic Innovations (ASU) in collaboration with Salt River Project (SRP). High resolution (3m) satellite imagery from Planet Labs
is used as input to the machine learning model, with rasterized lidar surveys (conducted by SRP) used as ground truth labels.

## Requirements
The scripts in this repository are in the form of Jupyter notebooks, run in a Python3 CUDA environment and based on the PyTorch framework. Modules from the following python packages are utilised, and these can be installed via pip or conda/mamba:
* [torch](https://pytorch.org/get-started/locally/)
* [torchvision](https://pytorch.org/vision/stable/index.html)
* [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
* [torchgeo](https://torchgeo.readthedocs.io/en/stable/)
* [numpy](https://numpy.org/install/)
* [matplotlib](https://matplotlib.org/stable/install/index.html)

## Components and Setup
### Scripts

1. [scripts/Training Pipeline.ipynb](<scripts/Training Pipeline.ipynb>):
  * This is the main training notebook, which is split into three major sections: data loading, model configuration, and the training loop. 
  * The dataset classes defined in this notebook are implemented as custom [RasterDatasets](https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html) from torchgeo. Please refer the documentation to understand how to configure the files to be loaded.
  * Configure any paths that are specified, i.e. for data, saving model checkpoints, or saving final trained model weights. 
2. [scripts/Eval Pipeline.ipynb](<scripts/Eval Pipeline.ipynb>):
  * This notebook is used for evaluating the model trained using the training script, and is split into four major sections: data loading, loading the trained model, visualizing predictions, computing evaluation metrics.
  * Like above, the dataset classes here are implemented as custom RasterDatasets from torchgeo and have to be configured accordingly, using appropriate paths when initializing the Dataset objects.
3. [scripts/backboned_unet](scripts/backboned_unet):
  * This module contains PyTorch implementations of the base architecture for the UNet model, imported when loading the model in the above scripts. 

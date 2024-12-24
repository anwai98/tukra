# tukra

Welcome to `tukra`.

This is a library composed of convenience scripts for evaluating (and training) popular deep learning-based methods for biomedical image segmentation.

## Installation:
- TODO
    - CellPose: works out of the box.
    - InstanSeg: works out of the box.
    - StarDist: training works only for CPU - need to investigate GPU-supported tensorflow installation.
    - nnUNet: works out of the box from the original repo.
    - BioMedParse: custom installation from https://github.com/anwai98/BiomedParse -> at `dev` branch

## Supported Models:

### Training (and Finetuning):
- CellPose
- CellPose 2
- StarDist
- nnUNet

### Inference:
- CellPose
- CellPose 2
- CellPose 3
- InstanSeg
- BioMedParse

## TODO: Planned Frameworks
- StarDist (inference)
- SplineDist
- OmniPose
- InstanSeg (the training pipeline is not wrapped in a function, but rather over one experiment script. looks difficult to inherit it out-of-the-box here)

## Supported Datasets:
> Thanks to <a href="https://github.com/constantinpape/torch-em">`torch-em`</a>
- LIVECell
- OrgaSegment
- DSB
- PanNuke

## TODO: Planned Dataset Integration
- PlantSeg
- MouseEmbryo
- DeepBacs
- Covid IF

## Supported Evaluation Metrics:
> Thanks to <a href="">`elf`</a>
- Segmentation Accuracy
    - Mean Segmentation Accuracy
    - Segmentation Accuracy over Intersection of Union (n)%
    - Dice Score

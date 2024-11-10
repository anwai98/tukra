# tukra

Welcome to `tukra`.

This is a library composed of convenience scripts for evaluating (and training) popular deep learning-based methods for biomedical image segmentation.

## Installation:
- TODO
    - CellPose: works out of the box.
    - InstanSeg: works out of the box.
    - StarDist: training works only for CPU - need to investigate GPU-supported tensorflow installation.

## Supported Models:

### Training (and Finetuning):
- CellPose
- CellPose 2
- StarDist

### Inference:
- CellPose
- CellPose 2
- CellPose 3
- InstanSeg

## TODO: Planned Frameworks
- nnUNet (support for instance segmentation)
- SplineDist
- OmniPose
- InstanSeg (the training pipeline is not wrapped in a function, but rather over one experiment script. looks difficult to inherit it out-of-the-box here)

## Supported Datasets:
> Thanks to <a href="https://github.com/constantinpape/torch-em">`torch-em`</a>
- LIVECell
- OrgaSegment
- DSB

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

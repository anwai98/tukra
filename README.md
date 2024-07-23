# tukra

Welcome to `tukra`.

This is a library composed of convenience scripts for evaluating (and training) popular deep learning-based methods for biomedical image segmentation.

## Supported Models:

### Training (and Finetuning):
- CellPose
- CellPose 2

### Inference:
- CellPose
- CellPose 2
- CellPose 3

## TODO: Planned Frameworks
- nnUNet (support for instance segmentation)
- StarDist
- SplineDist
- OmniPose

## Supported Datasets:
> Thanks to <a href="https://github.com/constantinpape/torch-em">`torch-em`</a>
- LIVECell
- OrgaSegment

## TODO: Planned Dataset Integration
- PlantSeg
- MouseEmbryo
- DeepBacs
- DSB
- Covid IF

## Supported Evaluation Metrics:
> Thanks to <a href="">`elf`</a>
- Segmentation Accuracy
    - Mean Segmentation Accuracy
    - Segmentation Accuracy over Intersection of Union (n)%

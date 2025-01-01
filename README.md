# tukra

Welcome to `tukra`.

This is a library composed of convenience scripts for evaluating (and training) popular deep learning-based methods for biomedical image segmentation.

## Quick Installation:
- Clone the repository: `https://github.com/anwai98/tukra.git`
- Enter the directory of repository: `cd tukra`
- Install `tukra` from source: `pip install -e .`

## Supported Models:

### Training (and/or Finetuning):
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
- InstanSeg

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
> Thanks to <a href="https://github.com/constantinpape/elf">`elf`</a>
- Segmentation Accuracy
    - Mean Segmentation Accuracy
    - Segmentation Accuracy over Intersection of Union (n)%
    - Dice Score

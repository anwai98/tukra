import os
from typing import Optional, List, Union

import numpy as np

import torch

from cellpose import denoise, models


MODEL_CHOICES = [
    "cyto",
    "cyto2",
    "cyto3",
    "livecell",
    "tissuenet",
    "nuclei",
    "livecell_cp3",
]

RESTORATION_CHOICES = [
    "denoise_cyto3",
    "deblur_cyto3",
    "upsample_cyto3",
    "denoise_nuclei",
    "deblur_nuclei",
    "upsample_nuclei",
]


def segment_using_cellpose(
    image: np.ndarray,
    model_choice: str,
    restoration_choice: Optional[str] = None,
    channels: List[int] = [0, 0],
    diameter: Optional[int] = None,
):
    """Supports inference on images using pretrained CellPose models.

    Args:
        image (np.ndarray): The input image
        model_choice (str): The choice of cellpose model. See 'MODEL_CHOICES' above.
        restoration_choice (str, None): The choice of image restoration cellpose model. See 'RESTORATION_CHOICE' above.
        channels (List[int]): The channel parameters to be used for inference.
        diameter (int, None): The diameter of the objects.

    Returns:
        masks (np.ndarray): The instance segmentation.
    """
    use_gpu = torch.cuda.is_available()

    if restoration_choice is None:
        if model_choice in ["cyto", "cyto2", "cyto3", "nuclei"]:  # generalist models
            model = models.Cellpose(gpu=use_gpu, model_type=model_choice)

            if diameter is None:
                diameter = 30  # the default choice

            masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=channels)

        elif model_choice in ["livecell", "livecell_cp3", "tissuenet"]:  # specialist models
            model = models.CellposeModel(gpu=use_gpu, model_type=model_choice)
            masks, flows, styles = model.eval(image, diameter=diameter, channels=channels)

        else:
            raise ValueError(f"{model_choice} is not supported in 'tukra'.")

    else:
        assert restoration_choice in RESTORATION_CHOICES, f"{restoration_choice} is not supported in 'tukra'."
        model = denoise.CellposeDenoiseModel(
            gpu=use_gpu, model_type=model_choice, restore_type=restoration_choice,
        )
        masks, flows, styles, denoised_images = model.eval(image, diameter=diameter, channels=channels)

    return masks


def segment_using_custom_cellpose(
    image: np.ndarray,
    checkpoint_path: Union[os.PathLike, str],
    channels: List[int] = [0, 0],
    diameter: Union[float, int] = 0,
    **kwargs
):
    """Supports inference on images using custom trained (or finetuned) CellPose models.

    Args:
        image (np.ndarray): The input image
        checkpoint_path (os.PathLike, str): The path where the trained model checkpoints are stored.
        channels (List[int]): The channel parameters to be used for inference.
        diameter (int, None): The diameter of the objects.

    Returns:
        masks (np.ndarray): The instance segmentation.
    """
    use_gpu = torch.cuda.is_available()

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=checkpoint_path)

    if diameter == 0:
        diameter = model.diam_labels

    masks, flows, styles = model.eval(image, diameter=diameter, channels=channels, **kwargs)

    return masks

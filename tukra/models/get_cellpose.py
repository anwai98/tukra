from typing import Optional, List

import numpy as np

import torch

from cellpose import denoise, models


MODEL_CHOICES = [
    "cyto",
    "cyto2",
    "cyto3",
    "livecell",
    "tissuenet",
    "nuclei"
]

RESTORATION_CHOICES = [
    "denoise_cyto3",
    "deblur_cyto3",
    "upsample_cyto3",
    "denoise_nuclei",
    "deblur_nuclei",
    "upsample_nuclei"
]


def segment_using_cellpose(
    image: np.ndarray,
    model_choice: str,
    restoration_choice: Optional[str] = None,
    channels: List[int, int] = [0, 0],
    diameter: Optional[int] = None,
):
    """Supports CellPose models.
    Arguments:
        image: The input image
        model_choice: The choice of cellpose model. See 'MODEL_CHOICES' above.
        restoration_choice: The choice of image restoration cellpose model. See 'RESTORATION_CHOICE' above.
        channels: TODO
        diameter: TODO
    """
    assert model_choice in MODEL_CHOICES, f"{model_choice} is not supported in 'tukra'."

    use_gpu = torch.cuda.is_available()

    if restoration_choice is None:
        if model_choice in ["cyto", "cyto2", "cyto3", "nuclei"]:  # specialist models
            model = models.Cellpose(gpu=True, model_type=model_choice)
            diameter = 30  # the default choice
        else:  # generalist models
            model = models.CellposeModel(gpu=use_gpu, model_type=model_choice)

        masks, flows, styles, diams = model.eval(
            image, diameter=diameter, channels=channels,
        )

    else:
        assert restoration_choice in RESTORATION_CHOICES, f"{restoration_choice} is not supported in 'tukra'."
        model = denoise.CellposeDenoiseModel(
            gpu=use_gpu, model_type=model_choice, restore_type=restoration_choice,
        )
        masks, flows, styles, denoised_images = model.eval(image, diameter=diameter, channels=channels)

    return masks

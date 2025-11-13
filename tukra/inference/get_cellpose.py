import os
from typing import Optional, List, Union, Tuple

import numpy as np

try:
    from cellpose import models
    _cellpose_is_installed = True
except ImportError:
    _cellpose_is_installed = False


MODEL_CHOICES = [
    "cyto",
    "cyto2",
    "cyto3",
    "livecell",
    "tissuenet",
    "nuclei",
    "livecell_cp3",
    "cpsam",
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
) -> np.ndarray:
    """Supports inference on images using pretrained CellPose models.

    Args:
        image: The input image
        model_choice: The choice of cellpose model. See 'MODEL_CHOICES' above.
        restoration_choice: The choice of image restoration cellpose model. See 'RESTORATION_CHOICE' above.
        channels: The channel parameters to be used for inference.
        diameter: The diameter of the objects.

    NOTE:
    1. For CellPose-SAM:
        a. You do not need to adjust the channels for histopathology BF images.
        b. For fluoroscence images, cytoplasn / membrane stain becomes channel 1, next is nuclear,
           and the third is 'None'.

    Returns:
        masks: The instance segmentation.
    """
    assert _cellpose_is_installed, "Please install 'cellpose'."

    import torch
    use_gpu = torch.cuda.is_available()

    if restoration_choice is None:
        if model_choice in ["cyto", "cyto2", "cyto3", "nuclei"]:  # generalist models
            model = models.Cellpose(gpu=use_gpu, model_type=model_choice)

            if diameter is None:
                diameter = 30  # the default choice

            masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=channels)

        elif model_choice in ["livecell", "livecell_cp3", "tissuenet", "cpsam"]:  # specialist models
            kwargs = {}
            if model_choice != "cpsam":  # Other model types need to be defined.
                kwargs["model_type"] = model_choice

            model = models.CellposeModel(gpu=use_gpu, **kwargs)
            masks, flows, styles = model.eval(image, diameter=diameter, channels=channels)

        else:
            raise ValueError(f"{model_choice} is not supported in 'tukra'.")

    else:
        from cellpose import denoise
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
    return_flows: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Supports inference on images using custom trained (or finetuned) CellPose models.

    Args:
        image: The input image.
        checkpoint_path: The path where the trained model checkpoints are stored.
        channels: The channel parameters to be used for inference.
        diameter: The diameter of the objects.
        return_flows: Whether to return the predicted masks, flows and styles.

    Returns:
        masks: The instance segmentation.
    """
    assert _cellpose_is_installed, "Please install 'cellpose'."

    import torch
    use_gpu = torch.cuda.is_available()

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=checkpoint_path)

    if diameter == 0:
        diameter = model.diam_labels

    masks, flows, styles = model.eval(image, diameter=diameter, channels=channels, **kwargs)

    if return_flows:
        return masks, flows, styles
    else:
        return masks

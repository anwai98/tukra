import os
from typing import Literal

import numpy as np

import torch

try:
    from instanseg import InstanSeg
except ImportError:
    InstanSeg = None


def segment_using_instanseg(
    image: np.ndarray,
    model_type: Literal["brightfield_nuclei", "fluorescence_nuclei_and_cells"],
    target: Literal["nuclei", "cells", "all_outputs"] = "all_outputs",
    scale: Literal["small", "medium"] = "small",
    verbosity: bool = True,
    **kwargs
) -> np.ndarray:
    """Supports inference on images using pretrained InstanSeg models.

    Args:
        image: The input image.
        model_type: The choice of instanseg model.
        target: The choice of targets to segment.
        scale: The scale of input images.
        verbosity: The choice of verbosity for model prediction.
        kwargs: Additional supported arguments for inference.

    Returns:
        The instance segmentation.
    """
    assert InstanSeg is not None, "Please install 'instanseg'."
    assert model_type in ["brightfield_nuclei", "fluorescence_nuclei_and_cells"]

    if image.ndim == 2:  # InstanSeg does not accept one channel images. Convert to RGB-style to avoid param mismatch.
        image = np.stack([image] * 3, axis=-1)
    
    bioimageio_path = os.environ.get("INSTANSEG_BIOIMAGEIO_PATH")
    if bioimageio_path:
        model_type = torch.jit.load(os.path.join(bioimageio_path, model_type, "instanseg.pt"))

    model = InstanSeg(model_type=model_type, verbosity=(1 if verbosity else 0))

    if scale == "small":
        labels, _ = model.eval_small_image(image=image, target=target, **kwargs)
    elif scale == "medium":  # enables tiling window based prediction
        labels, _ = model.eval_medium_image(image=image, target=target, **kwargs)
    else:
        raise ValueError(f"'{scale}' is not a valid prediction scale-mode.")

    labels = labels.squeeze().numpy()

    return labels

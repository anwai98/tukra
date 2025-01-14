from typing import Literal

import numpy as np

try:
    from instanseg import InstanSeg
except ImportError:
    InstanSeg = None
import os
import torch

def segment_using_instanseg(
    image: np.ndarray,
    model_type: Literal["brightfield_nuclei", "fluorescence_nuclei_and_cells"],
    target: Literal["nuclei", "cells", "all_outputs"] = "all_outputs",
    scale: Literal["small", "medium"] = "small",
    **kwargs
) -> np.ndarray:
    """Supports inference on images using pretrained InstanSeg models.

    Args:
        image: The input image.
        model_type: The choice of instanseg model.
        target: The choice of targets to segment.
        scale: The scale of input images.
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
        model = InstanSeg(torch.jit.load(os.path.join(bioimageio_path, model_type, "instanseg.pt")), verbosity=0)
    else: 
        model = InstanSeg(model_type, verbosity=0)
        print("InstanSeg will be downloaded.")
    
    if scale == "small":
        labels, _ = model.eval_small_image(image=image, target=target, **kwargs)
    elif scale == "medium":  # enables tiling window based prediction
        labels, _ = model.eval_medium_image(image=image, target=target, **kwargs)
    else:
        raise ValueError(f"'{scale}' is not a valid prediction scale-mode.")

    labels = labels.squeeze().numpy()

    return labels

from typing import Literal

import numpy as np

try:
    from instanseg import InstanSeg
except ImportError:
    InstanSeg = None


def segment_using_instanseg(
    image: np.ndarray,
    model_type: Literal["brightfield_nuclei", "fluorescence_nuclei_and_cells"],
    target: Literal["nuclei", "cells", "all_outputs"] = "all_outputs",
    scale: Literal["small", "medium"] = "small",
    **kwargs
):
    """
    """
    assert InstanSeg is not None, "Please install 'instanseg'."
    assert model_type in ["brightfield_nuclei", "fluorescence_nuclei_and_cells"]

    if image.ndim == 2:  # InstanSeg does not accept one channel images. Convert to RGB-style to avoid param mismatch.
        image = np.stack([image] * 3, axis=-1)

    model = InstanSeg("brightfield_nuclei", verbosity=1)

    if scale == "small":
        labels, _ = model.eval_small_image(image=image, target=target, **kwargs)
    elif scale == "medium":  # enables tiling window based prediction
        labels, _ = model.eval_medium_image(image=image, target=target, **kwargs)
    else:
        raise ValueError(f"'{scale}' is not a valid prediction scale-mode.")

    labels = labels.squeeze().numpy()

    return labels

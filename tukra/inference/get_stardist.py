from typing import Literal, Optional

import numpy as np

try:
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D
    _stardist_is_installed = True
except ImportError:
    _stardist_is_installed = False


def segment_using_stardist(
    image: np.ndarray,
    model_name: Optional[Literal["2D_demo", "2D_paper_dsb2018", "2D_versatile_fluo", "2D_versatile_he"]],
) -> np.ndarray:
    """Supports inference on images using pretrained StarDist models.

    Args:
        image: The input image.
        model_name: The choice of name for pretrained model.

    Returns:
        seg: The instance segmentation.
    """
    assert _stardist_is_installed, "Please install 'stardist'."

    scale = 1  # The scale factor for input image.
    if model_name is None:
        # TODO: Add supports for custom trained models.
        model = StarDist2D(None, name='stardist', basedir='models')
    else:
        model = StarDist2D.from_pretrained(model_name)

    if model is None:
        raise RuntimeError("The model should not be 'None'. Something went wrong.")

    if image.ndim == 2:  # We leave grayscale images as it is.
        pass
    elif image.ndim == 3 and image.shape[-1] == 3:  # For RGB data, we average the channels.
        image = np.mean(image, axis=-1)

    input_ = normalize(image, 1.0, 99.8)
    seg, _ = model.predict_instances(input_, scale=scale)

    return seg

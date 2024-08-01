import os
from tqdm import tqdm
from typing import Optional, Union, List, Tuple

import numpy as np

from tukra.io import read_image

from csbdeep.utils import normalize

from stardist.models import Config2D, StarDist2D
from stardist import gputools_available, fill_label_holes, calculate_extents


def run_stardist_training(
    train_image_paths: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    train_label_paths: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    val_image_paths: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    val_label_paths: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    model_name: str = "stardist",
    save_root: Optional[Union[str, os.PathLike]] = None,
    pretrained_backbone: Optional[str] = None,
    image_extension: str = ".tif",
    n_channels: Optional[int] = None,
    n_rays: int = 32,
    use_gpu: bool = False,
    grid: Tuple[int, int] = (2, 2),
):
    """
    """
    use_gpu = (use_gpu and gputools_available())

    train_images = [read_image(path, image_extension) for path in train_image_paths]
    train_labels = [read_image(path, image_extension) for path in train_label_paths]
    val_images = [read_image(path, image_extension) for path in val_image_paths]
    val_labels = [read_image(path, image_extension) for path in val_label_paths]

    if n_channels is None:
        n_channels = 1 if train_images[0].ndim == 2 else train_images[0].shape[-1]

    axis_norm = (0, 1)   # normalize channels independently
    # axis_norm = (0, 1, 2) # normalize channels jointly

    train_images = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(train_images)]
    train_labels = [fill_label_holes(y) for y in tqdm(train_labels)]

    val_images = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(val_images)]
    val_labels = [fill_label_holes(y) for y in tqdm(val_labels)]

    configuration = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channels,
    )
    print(configuration)

    model_kwargs = {}
    if save_root is not None:
        model_kwargs["base_dir"] = save_root

    # Initialize the model
    if pretrained_backbone is None:
        model = StarDist2D(configuration, name=model_name, **model_kwargs)
    else:
        assert pretrained_backbone in ["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018", "2D_demo"]
        model = StarDist2D.from_pretrained(pretrained_backbone)

    # Verification step of the objects w.r.t. the field of view set.
    median_size = calculate_extents(list(train_labels), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size: {median_size}")
    print(f"network field of view : {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    # Training the model.
    model.train(
        X=train_images,
        Y=train_labels,
        validation_data=(val_images, val_labels),
        augmenter=None,
    )

    # Threshold optimization
    model.optimize_thresholds(val_images, val_labels)

import os
from glob import glob
from typing import Union

from torch_em.data import datasets


def fetch_livecell_paths(
    data_dir: Union[os.PathLike, str],
    split: str = "test"
):
    """Returns the paths to all the livecell images per split.
    """
    assert split == "test", "We only support the test split at the moment."

    if not os.path.exists(data_dir):
        print("The dataset was not found. 'tukra' will download the dataset in the expected structure.")
        datasets.get_livecell_dataset(path=data_dir, patch_shape=(512, 512), download=True)

    all_image_paths = glob(os.path.join(data_dir, "images", "livecell_test_images", "*.tif"))
    all_gt_paths = glob(os.path.join(data_dir, "annotations", "livecell_test_images", "*", "*.tif"))

    return sorted(all_image_paths), sorted(all_gt_paths)

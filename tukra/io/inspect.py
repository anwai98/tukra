import os
from glob import glob

import numpy as np

from ..io import read_image


def tukra_inspect(input_paths, unique_ids=False, keys=None):
    """Inspects the image data located at the provided filepath.

    Args:
        input_paths: List of filepath to the image data.
        keys: List of hierarchy names for container-based data structures.
    """
    for input_path, key in zip(input_paths, keys):
        image = read_image(input_path=input_path, key=key)

        print()
        print(f"Inspecting information about the image: '{os.path.basename(input_path)}'")
        print(f"Shape of image: '{image.shape}'")
        print(f"Data type of image: '{image.dtype}'")
        print(f"Intensity range: '{np.min(image)}' to '{np.max(image)}'")
        print(f"Mean and standard deviation of intensity values: '{np.mean(image)}', '{np.std(image)}'")
        if unique_ids:
            print(f"Unique IDS: '{np.unique(image)}'")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect images.")
    parser.add_argument(
        "input_path", nargs="+", type=str,
        help="Expects a filepath or a sequence of filepaths to the image data. Currently supports all file "
        "formats supported by 'tukra' (eg. nifty, mha, dicom, nrrd, imageio-supported formats: tif, png, etc., "
        "elf-supported formats): hdf5, zarr, n5, mrc, knossos."
    )
    parser.add_argument(
        "-k", "--key", nargs="*", type=str, default=None,
        help="The key for opening data with 'elf.io.open_file'. This is the hierarchy name for a hdf5 or "
        "zarr container, for an image stack it is a wild-card, e.g. '*.png' and for mrc it is 'data'."
    )
    parser.add_argument(
        "-u", "--unique_ids", action="store_true",
        help="Whether to return unique ids for the entire array under inspection."
    )
    args = parser.parse_args()

    _input_paths = args.input_path
    _keys = args.key
    if _keys is None:
        _keys = [None] * len(_input_paths)
    else:
        if len(_keys) > len(_input_paths):  # This is the case where multiple keys are provided for one input path
            assert len(_input_paths) == 1, "Visualizing with multiple keys works only for one input path at the moment."
            _input_paths = _input_paths * len(_keys)

    # Check for filepath search patterns (eg. "*")
    for ipath in _input_paths:
        if "*" in ipath:
            _input_paths.extend(glob(os.path.join(ipath)))

    tukra_inspect(input_paths=_input_paths, unique_ids=args.unique_ids, keys=_keys)


if __name__ == "__main__":
    main()

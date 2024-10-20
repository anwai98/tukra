import os
from glob import glob

from ..io import read_image


def tukra_viewer(input_paths, keys=None, channels_first=False):
    """Opens the image data located at the provided filepath.

    Args:
        input_paths: List of filepath to the image data.
        keys: List of hierarchy names for container-based data structures.
        channels_first: Whether to make channels first in 3d volumes.
    """
    import napari
    _viewer = napari.Viewer()

    for input_path, key in zip(input_paths, keys):
        image = read_image(input_path=input_path, key=key)

        if channels_first:
            assert image.ndim == 3,  "The channels first feature is supported for 3d volumes only."
            image = image.transpose(2, 0, 1)

        _viewer.add_image(image)

    napari.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_path", nargs="+", type=str, required=True, help="")
    parser.add_argument("-k", "--key", nargs="*", type=str, default=None, help="")
    parser.add_argument("--ensure_channels_first", action="store_true", help="")
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

    tukra_viewer(input_paths=_input_paths, keys=_keys, channels_first=args.ensure_channels_first)


if __name__ == "__main__":
    main()

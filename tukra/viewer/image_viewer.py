from ..io import read_image


def tukra_viewer(input_paths, keys=None):
    """Opens the image data located at the provided filepath.

    Args:
        input_path: Filepath to the image data.
        key: The hierarchy name for container-based data structures.
    """
    import napari
    _viewer = napari.Viewer()

    for input_path, key in zip(input_paths, keys):
        image = read_image(input_path=input_path, key=key)
        _viewer.add_image(image)

    napari.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_path", nargs="+", type=str, required=True, help="")
    parser.add_argument("-k", "--key", nargs="*", type=str, default=None, help="")
    args = parser.parse_args()

    _input_paths = args.input_path
    _keys = args.key
    if _keys is None:
        _keys = [None] * len(_input_paths)

    tukra_viewer(input_paths=_input_paths, keys=_keys)


if __name__ == "__main__":
    main()

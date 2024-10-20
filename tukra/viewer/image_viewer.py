from ..io import read_image


def tukra_viewer(input_path, key=None):
    """Opens the image data located at the provided filepath.

    Args:
        input_path: Filepath to the image data.
        key: The hierarchy name for container-based data structures.
    """
    image = read_image(input_path=input_path, key=key)

    import napari
    v = napari.Viewer()
    v.add_image(image)
    napari.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="")
    parser.add_argument("-k", "--key", type=str, default=None, help="")
    args = parser.parse_args()

    tukra_viewer(input_path=args.input_path, key=args.key)


if __name__ == "__main__":
    main()

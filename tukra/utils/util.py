import os


def _all_imageio_formats():
    import imageio
    all_formats = [imageio.formats[f"{fmt.name}"].extensions for fmt in imageio.formats]
    return all_formats


def read_image(input_path, return_original_manifest=False):
    assert os.path.exists(input_path)

    extension = os.path.splitext(input_path)[-1]

    if extension == ".nii" or extension == ".nii.gz":
        import nibabel as nib
        inputs = nib.load(input_path)
        input_array = inputs.get_fdata()

    elif extension == ".mha":
        import SimpleITK as sitk
        inputs = sitk.ReadImage(input_path)
        input_array = sitk.GetArrayFromImage(inputs)

    elif extension in _all_imageio_formats():
        import imageio.v3 as imageio
        inputs = imageio.imread(input_path)
        input_array = inputs

    else:
        raise ValueError(f"'{extension}' is not a supported extension to read images in 'tukra'.")

    if return_original_manifest:
        return input_array, inputs
    else:
        return input_array

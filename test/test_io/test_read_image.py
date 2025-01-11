import os
import unittest
import urllib.request

import numpy as np


class TestReadImage(unittest.TestCase):

    def test_tif_read_image(self):
        expected_shape = (27, 128, 128)

        # Download the file on the fly.
        url = "https://github.com/tlnagy/exampletiffs/raw/master/mri.tif"
        image_path = "mri.tif"
        urllib.request.urlretrieve(url, image_path)
        print(f"The sample tif image has been download at: {image_path}")

        # Read the image.
        from tukra.io import read_image
        image = read_image(image_path)

        # Check for valid shape expectation.
        self.assertEqual(image.shape, expected_shape)
        self.assertIsInstance(image, np.ndarray)

        os.remove(image_path)


if __name__ == "__main__":
    unittest.main()

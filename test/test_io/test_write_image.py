import os
import unittest

import numpy as np


class TestWriteImage(unittest.TestCase):

    def test_tif_write_image(self):
        desired_shape = (256, 256)

        image = np.zeros(desired_shape)
        image_path = "test_image.tif"

        # Write the array.
        from tukra.io import write_image
        write_image(image_path, image)

        # Test by reading back the array
        from tukra.io import read_image
        new_image = read_image(image_path)

        self.assertTrue(np.array_equal(new_image, image))

        os.remove(image_path)


if __name__ == "__main__":
    unittest.main()

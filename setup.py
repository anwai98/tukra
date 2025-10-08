#!/usr/bin/env python

import runpy
from setuptools import find_packages, setup


__version__ = runpy.run_path("tukra/__version__.py")["__version__"]


setup(
    name='tukra',
    version=__version__,
    author='Anwai Archit',
    author_email='anwai.archit@uni-goettingen.de',
    url='https://github.com/anwai98/tukra',
    packages=find_packages(include=['tukra', 'tukra.*']),
    license="MIT",
    entry_points={
        "console_scripts": [
            "tukra.viewer = tukra.viewer.image_viewer:main",
            "tukra.inspect = tukra.io.inspect:main"
        ]
    }
)

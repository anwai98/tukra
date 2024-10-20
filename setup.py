#!/usr/bin/env python

from distutils.core import setup

setup(
    name='tukra',
    version='0.0.1',
    description='Functionality for evaluating deep learning-based segmentation methods.',
    author='Anwai Archit',
    author_email='anwai.archit@uni-goettingen.de',
    url='https://user.informatik.uni-goettingen.de/~pape41/',
    packages=['tukra'],
    entry_points={
        "console_scripts": [
            "tukra.viewer = tukra.viewer.image_viewer:main"
        ]
    }
)

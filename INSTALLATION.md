# Installation (from source):
- Clone the repository: `https://github.com/anwai98/tukra.git`
- Enter the directory of repository: `cd tukra`
- Create a new environment: `conda env create -n tukra-dev python=3.11`
- Activate the environment: `conda activate tukra-dev`
- Install `tukra` from source: `pip install -e .`

> NOTE: Below mentioned are the installation scripts for each supported frameworks.

## Install `StarDist`:

See the original repository [`StarDist`](https://github.com/stardist/stardist) for details.

> Comment: Install `TensorFlow` (with CUDA support) and `StarDist` both from `pip`.

## Install `CellPose`:

See the original repository [`CellPose`](https://github.com/MouseLand/cellpose) for details.

> Comment: Install `PyTorch` and `CellPose` both preferably from `pip`.

## Install `nnUNet`:

See the original repository [`nnUNet`](https://github.com/MIC-DKFZ/nnUNet) for details.

> Comment: Install `PyTorch` first and then `nnUNet` from source.

## Install `InstanSeg`:

See the original repository [`InstanSeg`](https://github.com/instanseg/instanseg) for details.

## Install `BioMedParse`:

- Clone the repository: `git clone https://github.com/anwai98/BiomedParse.git`
- Access the repository: `cd BiomedParse`
- Checkout to the `dev` branch: `git checkout dev`
- Create environment: `conda create -n biomedparse python=3.9.19`
- Activate environment: `conda activate biomedparse`
- Finally, run the shell script located at https://github.com/anwai98/BiomedParse/blob/dev/biomedparse_setup.sh to install all relevant packages:
    - Run the shell script: `bash biomedparse_setup.sh <CONDA_CMD>`, where `CONDA_CMD` is the conda prefix used for installing all relevant packages (eg. I use `micromamba` to install all packages)

    **OR**

    - Bootstrapped Installation:
    - Install PyTorch: `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
    - Update GCC in the environment: `conda install -c conda-forge gxx_linux-64=12` (for `detectron`'s installation)
        - Set compilers locally:
            - `export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc`
            - `export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++`
            - Verify the installation: `$CC --version`
    - Install additional packages for MPI compilation: `conda install -c conda-forge mpi4py openmpi`
    - Install all relevant requirements: `pip install -r assets/requirements/requirements.txt`
    - Install `torch_em`: `conda install -c conda-forge torch_em`.
    - Upgrade `tifffile`: `pip install --upgrade tifffile`.



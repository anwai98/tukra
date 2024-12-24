## Steps to install `BioMedParse`:

1. Clone the repository: `git clone https://github.com/anwai98/BiomedParse.git`
2. Access the repository: `cd BiomedParse`
3. Checkout to the `dev` branch: `git checkout dev`
4. Create environment: `conda create -n biomedparse python=3.9.19`
5. Activate environment: `conda activate biomedparse`

### Straight-forward Workflow
1. Run `bash ./biomedparse_setup.sh micromamba` to install all modules as expected.

### Bootstrapped Installation

1. Install PyTorch: `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
2. Update GCC in the environment: `conda install -c conda-forge gxx_linux-64=12` (for `detectron`'s installation)
    - Set compilers locally:
        - `export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc`
        - `export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++`
        - Verify the installation: `$CC --version`
3. Install additional packages for MPI compilation: `conda install -c conda-forge mpi4py openmpi`
4. Install all relevant requirements: `pip install -r assets/requirements/requirements.txt`
5. Install `torch_em`: `conda install -c conda-forge torch_em`.
6. Upgrade `tifffile`: `pip install --upgrade tifffile`.

#!/bin/bash
set -ex

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

cd ~

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

source ~/.bashrc

pip install matplotlib \
  jupyter \
  onnx \
  numpy \
  pytest \
  wheel \
  packaging \
  sklearn \
  sympy==1.1.1 \
  transformers==v2.10.0 \
  py-spy

pip install -r $SCRIPT_DIR/requirements.txt

# Install pytorch from source because the binary distribution from pip doesn't support all CUDA drivers.
# Following instructions here: https://github.com/pytorch/pytorch#from-source
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda101

# Note, you must use git for the source due to submodules. The release tarball doesn't work.
# releast 1.6.0 was failing to build so we just use master, whatever
git clone https://github.com/pytorch/pytorch.git
pushd pytorch

git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# Compilation took 3 hours and then crashed while linking because the binary
# was too big, this should speed it up and reduce binary size at the expense of
# generality. 3.5 is for the agpu1 machine. Unfortunately, there's no easy
# way to detect the compute capability of your device which would be the right
# way to do this.
export TORCH_CUDA_ARCH_LIST=3.5

python setup.py install

popd

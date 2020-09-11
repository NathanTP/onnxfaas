#!/bin/bash
set -ex

cd ~

apt-get update
apt-get install -y wget
# apt-get install -y vim wget libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
#
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init

source ~/.bashrc

pip install onnxruntime-gpu matplotlib jupyter onnx


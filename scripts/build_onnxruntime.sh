#!/bin/bash
set -e

cd ../onnxruntime

_CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)

./build.sh \
    --config Debug \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda \
    --build_wheel \
    --parallel \
    --skip_tests

pip install --no-deps --upgrade --force-reinstall build/Linux/Debug/dist/onnxruntime_gpu-1.4.0-cp38-cp38-linux_x86_64.whl

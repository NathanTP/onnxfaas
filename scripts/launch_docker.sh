#!/bin/bash
# Need hard to remember options to docker to forward 8888 (for jupyter), enable GPUs, and to mount everything we need
set -e

docker run -it --rm --gpus=all -v $(pwd)/../:/host -p 8888:8888 --expose 8888 onnxfaas bash

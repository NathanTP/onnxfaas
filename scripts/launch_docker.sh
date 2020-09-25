#!/bin/bash
# Need hard to remember options to docker to forward 8888 (for jupyter), enable GPUs, and to mount everything we need
set -e

if [[ $# == 1 ]]; then
  if [[ $1 == "build" ]]; then
    docker run -it --rm --gpus=all --privileged -v $(pwd)/../:/host onnxfaas:build bash
  elif [[ $1 == "deploy" ]]; then
    # the port forward is for jupyter
    docker run -it --rm --gpus=all --privileged -v $(pwd)/../:/host -p 8888:8888 --expose 8888 onnxfaas:deploy bash
  else
    docker run -it --rm --gpus=all --privileged -v $(pwd)/../:/host $1 bash
  fi
fi

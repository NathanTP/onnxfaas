#!/bin/bash
set -e

DOCKERARGS="--build-arg USER=$USER --build-arg UID=$(id -u) --build-arg GID=$(id -g)"
echo $DOCKERARGS

pushd ../docker
docker build $DOCKERARGS --tag onnxfaas:deploy -f ../docker/Dockerfile.deploy .
# docker build $DOCKERARGS --tag onnxfaas:build -f ../docker/Dockerfile.build .
popd

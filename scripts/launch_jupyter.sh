#!/bin/bash
# Jupyter is touchy on Docker, you have to launch it like this

set -e

jupyter notebook --ip=0.0.0.0 --allow-root

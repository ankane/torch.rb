#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/libtorch/$LIBTORCH_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION%2Bcpu.zip
  unzip libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION+cpu.zip
  mv libtorch $CACHE_DIR
else
  echo "LibTorch cached"
fi

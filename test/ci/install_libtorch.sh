#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/libtorch/$LIBTORCH_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  if [ -n "$LIBTORCH_GPU" ]; then
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION+cpu.zip
  else
    wget https://download.pytorch.org/libtorch/$LIBTORCH_GPU/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION.zip
    unzip libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION.zip
  fi
  mv libtorch $CACHE_DIR
else
  echo "LibTorch cached"
fi

#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/libtorch/$LIBTORCH_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION.zip
  unzip libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION.zip
  mv libtorch $CACHE_DIR
else
  echo "LibTorch cached"
fi

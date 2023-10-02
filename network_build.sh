#!/bin/bash

set -e

rm -rf build
mkdir build
cd build
cmake -DVICON=ON ..
make -j8

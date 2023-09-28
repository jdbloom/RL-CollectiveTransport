#!/bin/bash

set -e

rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -VICON=ON ..
make -j8

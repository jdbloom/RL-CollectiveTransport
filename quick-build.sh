#!/bin/bash

set -e

rm -rf build
mkdir build
cd build
cmake -DCLUSTER=OFF -DCMAKE_BUILD_TYPE=Release ..
make

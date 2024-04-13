#!/bin/bash

set -e
cd ..
rm -rf build
mkdir build
cd build
cmake -DCLUSTER=ON -DCMAKE_BUILD_TYPE=Release ..
make

#!/bin/sh
set -e

./configure --tests --benchmarks --examples
# make
make -j 8
make test

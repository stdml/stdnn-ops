#!/bin/sh
set -e

./configure --tests --benchmarks --examples
make -j 8
make test

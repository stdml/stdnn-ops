#!/bin/sh
set -e

flags() {
    echo --use-blas
    echo --tests
    echo --benchmarks
    echo --examples
}

./configure $(flags)

# make
make -j 8
make test

#!/bin/sh
set -e

flags() {
    echo --use-blas
    # echo --tests
    # echo --benchmarks
    echo --examples
    echo --with-blas=$HOME/local/openblas
}

./configure $(flags)
# exit

make
# make -j 8
make test

export STD_TRACER_REPORT_STDOUT=0
./bin/example-train-mnist-slp

#!/bin/sh
set -e

flags() {
    echo --tests
    echo --benchmarks
    echo --examples
    # echo --use-blas
    # echo --with-blas=$HOME/local/openblas
}

rebuild() {
    ./configure $(flags)
    # exit

    make -j $(nproc)
    make test

    # make test-softmax
    # ./bin/test-softmax
}

rebuild
# export STD_TRACER_REPORT_STDOUT=0
# ./bin/example-train-mnist-slp

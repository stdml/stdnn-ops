#!/bin/sh
set -e

./configure --prefix=$HOME/local
make install

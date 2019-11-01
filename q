#!/bin/sh
set -e

./configure --lib --examples --extern
make

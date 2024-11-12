#!/bin/bash -x
export MKLROOT=${PREFIX}
# $PYTHON setup.py config build install --old-and-unmanageable
$PYTHON -m pip install . --no-deps -vv

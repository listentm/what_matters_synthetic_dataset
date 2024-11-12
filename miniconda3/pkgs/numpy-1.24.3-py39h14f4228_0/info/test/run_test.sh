

set -ex



f2py -h
python -c "import numpy; numpy.show_config()"
export OPENBLAS_NUM_THREADS=1
pytest -vvv --pyargs numpy -k "not (_not_a_real_test or test_overrides)" --durations=0
exit 0

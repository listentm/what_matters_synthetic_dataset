

set -ex



pip check
pytest tests/ -vvv --ignore=tests/test_schema.py --ignore=tests/test_elevation.py
exit 0

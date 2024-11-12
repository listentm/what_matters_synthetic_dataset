set MKLROOT=%PREFIX%
@REM %PYTHON% setup.py config install --old-and-unmanageable
%PYTHON% -m pip install . --no-deps -vv
if errorlevel 1 exit 1

:: site.cfg should not be defined here.  It is provided by blas devel packages (either mkl-devel or openblas-devel)

COPY %PREFIX%\site.cfg site.cfg

%PYTHON% -m pip install --no-deps --ignore-installed --no-build-isolation -v .
if errorlevel 1 exit 1

XCOPY %RECIPE_DIR%\f2py.bat %SCRIPTS% /s /e
if errorlevel 1 exit 1

del %SCRIPTS%\f2py.exe
if errorlevel 1 exit 1
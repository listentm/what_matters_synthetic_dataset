setlocal EnableDelayedExpansion
echo on

:: brand Python with conda-forge startup message
if "%CONDA_FORGE%"=="yes" (
  %SYS_PYTHON% %RECIPE_DIR%\brand_python.py
  if errorlevel 1 exit 1
)

:: Compile python, extensions and external libraries
if "%ARCH%"=="64" (
   set PLATFORM=x64
   set VC_PATH=x64
   set BUILD_PATH=amd64
) else (
   set PLATFORM=Win32
   set VC_PATH=x86
   set BUILD_PATH=win32
)

set PY_VER=39

set "OPENSSL_DIR=%LIBRARY_PREFIX%"
set "SQLITE3_DIR=%LIBRARY_PREFIX%"
for /f "usebackq delims=" %%i in (`conda list -p %PREFIX% sqlite --no-show-channel-urls --json ^| findstr "version"`) do set SQLITE3_VERSION_LINE=%%i
for /f "tokens=2 delims==/ " %%i IN ('echo %SQLITE3_VERSION_LINE%') do (set SQLITE3_VERSION=%%~i)
echo SQLITE3_VERSION detected as %SQLITE3_VERSION%

if "%PY_INTERP_DEBUG%"=="yes" (
  set CONFIG=-d
  set _D=_d
) else (
  set CONFIG=
  set _D=
)


if "%DEBUG_C%"=="yes" (
  set PGO=
) else (
  set PGO=--pgo
)

:: AP doesn't support PGO atm?
if "%CONDA_FORGE%"=="yes" (
  set PGO=
)
:: PGO is not yet working for Python 3.9:
:: (CopyPGCFiles target) ->
::   C:\opt\conda\conda-bld\python-split-3.9.0_7\work\PCbuild\pyproject.props(165,5): error : PGO run did not succeed (no python39!*.pgc files) and there is no data to merge [C:\opt\conda\conda-bld\python-split-3.9.0_7\work\PCbuild\pythoncore.vcxproj]
:: set PGO=

cd PCbuild

:: Twice because:
:: error : importlib_zipimport.h updated. You will need to rebuild pythoncore to see the changes.
:: call build.bat %PGO% %CONFIG% -m -e -v -p %PLATFORM%
call build.bat %PGO% %CONFIG% -m -e -v -p %PLATFORM%
if errorlevel 1 exit 1
cd ..

:: Populate the root package directory
for %%x in (python%PY_VER%%_D%.dll python3%_D%.dll python%_D%.exe pythonw%_D%.exe venvlauncher%_D%.exe venvwlauncher%_D%.exe) do (
  if exist %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x (
    copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x %PREFIX%
  ) else (
    echo "WARNING :: %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x does not exist"
  )
)

for %%x in (python%_D%.pdb python%PY_VER%%_D%.pdb pythonw%_D%.pdb) do (
  if exist %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x (
    copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x %PREFIX%
  ) else (
    echo "WARNING :: %SRC_DIR%\PCbuild\%BUILD_PATH%\%%x does not exist"
  )
)

copy %SRC_DIR%\LICENSE %PREFIX%\LICENSE_PYTHON.txt
if errorlevel 1 exit 1

:: Populate the DLLs directory
mkdir %PREFIX%\DLLs
xcopy /s /y %SRC_DIR%\PCBuild\%BUILD_PATH%\*.pyd %PREFIX%\DLLs\
if errorlevel 1 exit 1
copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\tcl86t.dll %PREFIX%\DLLs\
if errorlevel 1 exit 1
copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\tk86t.dll %PREFIX%\DLLs\
if errorlevel 1 exit 1
copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\libffi-7.dll %PREFIX%\DLLs\
if errorlevel 1 exit 1

copy /Y %SRC_DIR%\PC\icons\py.ico %PREFIX%\DLLs\
if errorlevel 1 exit 1
copy /Y %SRC_DIR%\PC\icons\pyc.ico %PREFIX%\DLLs\
if errorlevel 1 exit 1

:: Populate the Tools directory
mkdir %PREFIX%\Tools
xcopy /s /y /i %SRC_DIR%\Tools\demo %PREFIX%\Tools\demo
if errorlevel 1 exit 1
xcopy /s /y /i %SRC_DIR%\Tools\i18n %PREFIX%\Tools\i18n
if errorlevel 1 exit 1
xcopy /s /y /i %SRC_DIR%\Tools\pynche %PREFIX%\Tools\pynche
if errorlevel 1 exit 1
xcopy /s /y /i %SRC_DIR%\Tools\scripts %PREFIX%\Tools\scripts
if errorlevel 1 exit 1

del %PREFIX%\Tools\demo\README
if errorlevel 1 exit 1
del %PREFIX%\Tools\pynche\README
if errorlevel 1 exit 1
del %PREFIX%\Tools\pynche\pynche
if errorlevel 1 exit 1
del %PREFIX%\Tools\scripts\README
if errorlevel 1 exit 1
del %PREFIX%\Tools\scripts\dutree.doc
if errorlevel 1 exit 1
del %PREFIX%\Tools\scripts\idle3
if errorlevel 1 exit 1

move /y %PREFIX%\Tools\scripts\2to3 %PREFIX%\Tools\scripts\2to3.py
if errorlevel 1 exit 1
move /y %PREFIX%\Tools\scripts\pydoc3 %PREFIX%\Tools\scripts\pydoc3.py
if errorlevel 1 exit 1

:: Populate the tcl directory
xcopy /s /y /i %SRC_DIR%\externals\tcltk-8.6.9.0\%BUILD_PATH%\lib %PREFIX%\tcl
if errorlevel 1 exit 1

:: Populate the include directory
xcopy /s /y %SRC_DIR%\Include %PREFIX%\include\
if errorlevel 1 exit 1

copy /Y %SRC_DIR%\PC\pyconfig.h %PREFIX%\include\
if errorlevel 1 exit 1

:: Populate the Scripts directory
if not exist %SCRIPTS% (mkdir %SCRIPTS%)
if errorlevel 1 exit 1

for %%x in (idle pydoc) do (
    copy /Y %SRC_DIR%\Tools\scripts\%%x3 %SCRIPTS%\%%x
    if errorlevel 1 exit 1
)

copy /Y %SRC_DIR%\Tools\scripts\2to3 %SCRIPTS%
if errorlevel 1 exit 1

:: Populate the libs directory
if not exist %PREFIX%\libs mkdir %PREFIX%\libs
if exist %SRC_DIR%\PCbuild\%BUILD_PATH%\python%PY_VER%%_D%.lib copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\python%PY_VER%%_D%.lib %PREFIX%\libs\
if errorlevel 1 exit 1
if exist %SRC_DIR%\PCbuild\%BUILD_PATH%\python3%_D%.lib copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\python3%_D%.lib %PREFIX%\libs\
if errorlevel 1 exit 1
if exist %SRC_DIR%\PCbuild\%BUILD_PATH%\_tkinter%_D%.lib copy /Y %SRC_DIR%\PCbuild\%BUILD_PATH%\_tkinter%_D%.lib %PREFIX%\libs\
if errorlevel 1 exit 1


:: Populate the Lib directory
del %PREFIX%\libs\libpython*.a
xcopy /s /y %SRC_DIR%\Lib %PREFIX%\Lib\
if errorlevel 1 exit 1

:: Remove test data to save space.
:: Though keep `support` as some things use that.
mkdir %PREFIX%\Lib\test_keep
if errorlevel 1 exit 1
move %PREFIX%\Lib\test\__init__.py %PREFIX%\Lib\test_keep\
if errorlevel 1 exit 1
move %PREFIX%\Lib\test\support %PREFIX%\Lib\test_keep\
if errorlevel 1 exit 1
rd /s /q %PREFIX%\Lib\test
if errorlevel 1 exit 1
move %PREFIX%\Lib\test_keep %PREFIX%\Lib\test
if errorlevel 1 exit 1

:: Remove some tests (unfortunate)
rd /s /q %PREFIX%\Lib\lib2to3\tests\
if errorlevel 1 exit 1

:: Remove PGO DLLs
if exist %PREFIX%\DLLs\instrumented (
  rd /s /q %PREFIX%\DLLs\instrumented\
  if errorlevel 1 exit 1
)

:: bytecode compile the standard library

rd /s /q %PREFIX%\Lib\lib2to3\tests\
if errorlevel 1 exit 1

:: We need our Python to be found!
if "%_D%" neq "" copy %PREFIX%\python%_D%.exe %PREFIX%\python.exe

%PREFIX%\python.exe -Wi %PREFIX%\Lib\compileall.py -f -q -x "bad_coding|badsyntax|py2_" %PREFIX%\Lib
if errorlevel 1 exit 1

:: Pickle lib2to3 Grammar
%PREFIX%\python.exe -m lib2to3 --help

:: Ensure that scripts are generated
:: https://github.com/conda-forge/python-feedstock/issues/384
echo ARCH is %ARCH%
echo ARCH is %ARCH%
echo ARCH is %ARCH%
echo ARCH is %ARCH%
%SYS_PREFIX%\python.exe %RECIPE_DIR%\fix_staged_scripts.py %ARCH%
if errorlevel 1 exit 1

:: Some quick tests for common failures
echo "Testing print() does not print: Hello"
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe -c "print()" 2>&1 | findstr /r /c:"Hello"
if %errorlevel% neq 1 exit /b 1

echo "Testing print('Hello') prints: Hello"
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe "print('Hello')" 2>&1 | findstr /r /c:"Hello"
if %errorlevel% neq 0 exit /b 1

echo "Testing import of os (no DLL needed) does not print: The specified module could not be found"
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe -v -c "import os" 2>&1
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe -v -c "import os" 2>&1 | findstr /r /c:"The specified module could not be found"
if %errorlevel% neq 1 exit /b 1

echo "Testing import of _sqlite3 (DLL located via PATH needed) does not print: The specified module could not be found"
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe -v -c "import _sqlite3" 2>&1
%CONDA_EXE% run -p %PREFIX% cd %PREFIX% & %PREFIX%\python.exe -v -c "import _sqlite3" 2>&1 | findstr /r /c:"The specified module could not be found"
if %errorlevel% neq 1 exit /b 1

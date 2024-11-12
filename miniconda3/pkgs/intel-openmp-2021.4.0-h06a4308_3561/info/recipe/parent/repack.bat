set "src=%SRC_DIR%\%PKG_NAME%"

@REM @REM The license files are no longer manually packed but are included via the meta.yaml
@REM if exist %src%\info\LICENSE.txt (
@REM     COPY %src%\info\LICENSE.txt %SRC_DIR%
@REM ) else (
@REM     COPY %SRC_DIR%\mkl\info\licenses\license.txt %SRC_DIR%
@REM )

robocopy /E "%src%" "%PREFIX%"
if %ERRORLEVEL% GEQ 8 exit 1

:: replace old info folder with our new regenerated one
rd /s /q %PREFIX%\info
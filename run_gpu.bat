@echo off
echo MaChain v1.6 - GPU Mode
echo ========================================
echo.

set LWJGL_CP=lib\lwjgl.jar;lib\lwjgl-opencl.jar;lib\lwjgl-natives-windows-x64.jar
set JAVA_CP=bin;%LWJGL_CP%

echo [LWJGL classpath: %LWJGL_CP%]
echo [Starting MaChain...]
echo.

java -cp "%JAVA_CP%" -Djava.library.path=lib --enable-native-access=ALL-UNNAMED machain.Main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Program exited with code: %ERRORLEVEL%
    pause
)

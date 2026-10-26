@echo off
echo MaChain v1.7 - 神经网络训练平台 (GPU加速+样本抓取+接口中心)
echo ===============================================
echo.

if not exist bin mkdir bin

set LWJGL_CP=lib\lwjgl.jar;lib\lwjgl-opencl.jar;lib\lwjgl-natives-windows-x64.jar

echo [Compiling...]
javac -encoding UTF-8 -cp "%LWJGL_CP%" -d bin ^
  src/machain/network/Network.java ^
  src/machain/network/OutputMode.java ^
  src/machain/network/DualStreamNetwork.java ^
  src/machain/network/EnhancedNetwork.java ^
  src/machain/data/DataSet.java ^
  src/machain/data/TextProcessor.java ^
  src/machain/data/SampleExtractor.java ^
  src/machain/data/NoiseFilter.java ^
  src/machain/api/ApiServer.java ^
  src/machain/io/SaveManager.java ^
  src/machain/io/SampleScraper.java ^
  src/machain/io/OutputExporter.java ^
  src/machain/generation/TextGenerator.java ^
  src/machain/core/TrainingCallback.java ^
  src/machain/core/TrainingResult.java ^
  src/machain/core/Trainer.java ^
  src/machain/core/DualTrainer.java ^
  src/machain/core/ConsoleTrainingCallback.java ^
  src/machain/config/Config.java ^
  src/machain/fuzzy/FuzzySkillBridge.java ^
  src/machain/output/OutputModule.java ^
  src/machain/output/CSVOutputModule.java ^
  src/machain/output/JSONOutputModule.java ^
  src/machain/output/OutputManager.java ^
  src/machain/gpu/CUDAContext.java ^
  src/machain/gpu/GPUMatrixOps.java ^
  src/machain/gpu/GPUAccelerator.java ^
  src/machain/ui/TrainingChart.java ^
  src/machain/ui/ConfusionMatrixPanel.java ^
  src/machain/ui/RangeSlider.java ^
  src/machain/ui/MainFrame.java ^
  src/machain/Main.java

if errorlevel 1 (
    echo Compile failed!
    pause
    exit /b 1
)
echo Compile OK.
echo.

echo [Starting MaChain v1.6...]
java -cp "bin;%LWJGL_CP%" -Djava.library.path=lib --enable-native-access=ALL-UNNAMED machain.Main

if errorlevel 1 (
    echo.
    echo Program exited with error.
    pause
)

package machain.output;

import machain.network.Network;
import machain.data.DataSet;
import machain.core.TrainingResult;

import java.io.File;
import java.util.*;

public class OutputManager {
    private final String outputDir;
    private final Map<String, OutputModule> modules = new LinkedHashMap<>();

    public OutputManager(String baseDir) {
        this.outputDir = baseDir + File.separator + "output";
        new File(outputDir).mkdirs();

        registerModule(new CSVOutputModule());
        registerModule(new JSONOutputModule());
    }

    public void registerModule(OutputModule module) {
        modules.put(module.getName().toLowerCase(), module);
    }

    public List<String> getAvailableModules() {
        return new ArrayList<>(modules.keySet());
    }

    public String exportTrainingResult(TrainingResult result, String format, String filename) throws Exception {
        OutputModule module = getModule(format);
        String ext = format.equalsIgnoreCase("json") ? ".json" : ".csv";
        String path = outputDir + File.separator + filename + ext;
        module.exportTrainingResult(result, path);
        return path;
    }

    public String exportPredictions(Network network, DataSet dataSet, String format, String filename) throws Exception {
        OutputModule module = getModule(format);
        String ext = format.equalsIgnoreCase("json") ? ".json" : ".csv";
        String path = outputDir + File.separator + filename + "_predictions" + ext;
        module.exportPredictions(network, dataSet, path);
        return path;
    }

    public String exportModelInfo(Network network, String format, String filename) throws Exception {
        OutputModule module = getModule(format);
        String ext = format.equalsIgnoreCase("json") ? ".json" : ".csv";
        String path = outputDir + File.separator + filename + "_info" + ext;
        module.exportModelInfo(network, path);
        return path;
    }

    private OutputModule getModule(String format) {
        OutputModule module = modules.get(format.toLowerCase());
        if (module == null) {
            throw new IllegalArgumentException("不支持的输出格式: " + format +
                    ". 可用格式: " + String.join(", ", modules.keySet()));
        }
        return module;
    }

    public String getOutputDir() {
        return outputDir;
    }
}

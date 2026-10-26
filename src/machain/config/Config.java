package machain.config;

import java.io.*;
import java.util.Properties;

public class Config implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final String DEFAULT_CONFIG_FILE = "machain.properties";

    private int[] layerSizes = {2, 4, 1};
    private String activation = "SIGMOID";
    private double learningRate = 0.01;
    private double momentum = 0.9;
    private int batchSize = 0;
    private double trainSplitRatio = 0.8;
    private boolean normalizeData = true;
    private String dataPath = "";
    private String saveDir = "saves";
    private String outputDir = "output";
    private int logInterval = 10;

    public Config() {}

    public void loadFromFile(String path) throws IOException {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(path)) {
            props.load(fis);
        }
        applyProperties(props);
    }

    public void saveToFile(String path) throws IOException {
        Properties props = toProperties();
        try (FileOutputStream fos = new FileOutputStream(path)) {
            props.store(fos, "MaChain Configuration");
        }
    }

    private void applyProperties(Properties props) {
        if (props.containsKey("network.layers")) {
            String[] parts = props.getProperty("network.layers").split(",");
            layerSizes = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                layerSizes[i] = Integer.parseInt(parts[i].trim());
            }
        }
        if (props.containsKey("network.activation"))
            activation = props.getProperty("network.activation");
        if (props.containsKey("training.learningRate"))
            learningRate = Double.parseDouble(props.getProperty("training.learningRate"));
        if (props.containsKey("training.momentum"))
            momentum = Double.parseDouble(props.getProperty("training.momentum"));
        if (props.containsKey("training.batchSize"))
            batchSize = Integer.parseInt(props.getProperty("training.batchSize"));
        if (props.containsKey("data.trainSplitRatio"))
            trainSplitRatio = Double.parseDouble(props.getProperty("data.trainSplitRatio"));
        if (props.containsKey("data.normalize"))
            normalizeData = Boolean.parseBoolean(props.getProperty("data.normalize"));
        if (props.containsKey("data.path"))
            dataPath = props.getProperty("data.path");
        if (props.containsKey("paths.saveDir"))
            saveDir = props.getProperty("paths.saveDir");
        if (props.containsKey("paths.outputDir"))
            outputDir = props.getProperty("paths.outputDir");
        if (props.containsKey("logging.interval"))
            logInterval = Integer.parseInt(props.getProperty("logging.interval"));
    }

    public Properties toProperties() {
        Properties props = new Properties();
        StringBuilder layers = new StringBuilder();
        for (int i = 0; i < layerSizes.length; i++) {
            if (i > 0) layers.append(",");
            layers.append(layerSizes[i]);
        }
        props.setProperty("network.layers", layers.toString());
        props.setProperty("network.activation", activation);
        props.setProperty("training.learningRate", String.valueOf(learningRate));
        props.setProperty("training.momentum", String.valueOf(momentum));
        props.setProperty("training.batchSize", String.valueOf(batchSize));
        props.setProperty("data.trainSplitRatio", String.valueOf(trainSplitRatio));
        props.setProperty("data.normalize", String.valueOf(normalizeData));
        props.setProperty("data.path", dataPath);
        props.setProperty("paths.saveDir", saveDir);
        props.setProperty("paths.outputDir", outputDir);
        props.setProperty("logging.interval", String.valueOf(logInterval));
        return props;
    }

    // Getters and setters
    public int[] getLayerSizes() { return layerSizes.clone(); }
    public void setLayerSizes(int[] sizes) { this.layerSizes = sizes.clone(); }
    public String getActivation() { return activation; }
    public void setActivation(String act) { this.activation = act; }
    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double lr) { this.learningRate = lr; }
    public double getMomentum() { return momentum; }
    public void setMomentum(double m) { this.momentum = m; }
    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int bs) { this.batchSize = bs; }
    public double getTrainSplitRatio() { return trainSplitRatio; }
    public void setTrainSplitRatio(double r) { this.trainSplitRatio = r; }
    public boolean isNormalizeData() { return normalizeData; }
    public void setNormalizeData(boolean n) { this.normalizeData = n; }
    public String getDataPath() { return dataPath; }
    public void setDataPath(String p) { this.dataPath = p; }
    public String getSaveDir() { return saveDir; }
    public void setSaveDir(String d) { this.saveDir = d; }
    public String getOutputDir() { return outputDir; }
    public void setOutputDir(String d) { this.outputDir = d; }
    public int getLogInterval() { return logInterval; }
    public void setLogInterval(int i) { this.logInterval = i; }
}

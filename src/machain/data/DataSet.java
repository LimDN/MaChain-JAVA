package machain.data;

import java.io.*;
import java.util.*;

public class DataSet implements Serializable {
    private static final long serialVersionUID = 1L;

    private double[][] inputs;
    private double[][] targets;
    private String[] featureNames;
    private String[] labelNames;
    private double[] inputMin, inputMax, inputMean, inputStd;
    private boolean normalized = false;
    private String sourceFile = "";

    public DataSet() {}

    public static DataSet loadCSV(String path, int inputCols, int targetCols, boolean hasHeader) throws IOException {
        DataSet ds = new DataSet();
        ds.sourceFile = path;
        List<double[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first && hasHeader) {
                    first = false;
                    String[] headers = line.split(",");
                    ds.featureNames = new String[inputCols];
                    ds.labelNames = new String[targetCols];
                    for (int i = 0; i < inputCols && i < headers.length; i++)
                        ds.featureNames[i] = headers[i].trim();
                    for (int i = 0; i < targetCols && (inputCols + i) < headers.length; i++)
                        ds.labelNames[i] = headers[inputCols + i].trim();
                    continue;
                }
                first = false;
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i].trim());
                }
                rows.add(row);
            }
        }

        int totalCols = inputCols + targetCols;
        ds.inputs = new double[rows.size()][inputCols];
        ds.targets = new double[rows.size()][targetCols];

        for (int r = 0; r < rows.size(); r++) {
            double[] row = rows.get(r);
            for (int c = 0; c < inputCols; c++)
                ds.inputs[r][c] = c < row.length ? row[c] : 0;
            for (int c = 0; c < targetCols; c++)
                ds.targets[r][c] = (inputCols + c) < row.length ? row[inputCols + c] : 0;
        }

        if (ds.featureNames == null) {
            ds.featureNames = new String[inputCols];
            ds.labelNames = new String[targetCols];
            for (int i = 0; i < inputCols; i++) ds.featureNames[i] = "x" + (i + 1);
            for (int i = 0; i < targetCols; i++) ds.labelNames[i] = "y" + (i + 1);
        }

        ds.computeStats();
        return ds;
    }

    public static DataSet loadClassificationCSV(String path, int inputCols, boolean hasHeader) throws IOException {
        DataSet ds = new DataSet();
        ds.sourceFile = path;
        List<double[]> inputRows = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        Map<String, Integer> labelMap = new LinkedHashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first && hasHeader) { first = false; continue; }
                first = false;
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split(",");
                double[] row = new double[inputCols];
                for (int i = 0; i < inputCols; i++) {
                    row[i] = Double.parseDouble(parts[i].trim());
                }
                inputRows.add(row);

                String label = parts[inputCols].trim();
                if (!labelMap.containsKey(label)) {
                    labelMap.put(label, labelMap.size());
                }
                labelList.add(labelMap.get(label));
            }
        }

        ds.inputs = new double[inputRows.size()][inputCols];
        int numClasses = labelMap.size();
        ds.targets = new double[labelList.size()][numClasses];

        for (int i = 0; i < inputRows.size(); i++) {
            ds.inputs[i] = inputRows.get(i);
            ds.targets[i][labelList.get(i)] = 1.0;
        }

        ds.featureNames = new String[inputCols];
        for (int i = 0; i < inputCols; i++) ds.featureNames[i] = "x" + (i + 1);
        ds.labelNames = labelMap.keySet().toArray(new String[0]);

        ds.computeStats();
        return ds;
    }

    public static DataSet loadTextCSV(String path, int textCol, int labelCol, boolean hasHeader,
                                       TextProcessor processor) throws IOException {
        DataSet ds = new DataSet();
        ds.sourceFile = path;
        List<String> texts = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        Map<String, Integer> labelMap = new LinkedHashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first && hasHeader) { first = false; continue; }
                first = false;
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split(",", -1);
                if (textCol >= parts.length || labelCol >= parts.length) continue;

                texts.add(parts[textCol].trim());
                String label = parts[labelCol].trim();
                if (!labelMap.containsKey(label)) {
                    labelMap.put(label, labelMap.size());
                }
                labels.add(labelMap.get(label));
            }
        }

        processor.buildVocabulary(texts);
        int vectorSize = processor.getVectorSize();

        ds.inputs = new double[texts.size()][vectorSize];
        ds.targets = new double[labels.size()][labelMap.size()];

        for (int i = 0; i < texts.size(); i++) {
            ds.inputs[i] = processor.vectorize(texts.get(i));
            ds.targets[i][labels.get(i)] = 1.0;
        }

        ds.featureNames = new String[vectorSize];
        for (int i = 0; i < vectorSize; i++) ds.featureNames[i] = "feat_" + i;
        ds.labelNames = labelMap.keySet().toArray(new String[0]);

        ds.computeStats();
        return ds;
    }

    public static DataSet fromArrays(double[][] inputs, double[][] targets,
                                      String[] featureNames, String[] labelNames) {
        DataSet ds = new DataSet();
        ds.inputs = inputs;
        ds.targets = targets;
        ds.featureNames = featureNames;
        ds.labelNames = labelNames;
        ds.computeStats();
        return ds;
    }

    private void computeStats() {
        int n = inputs.length;
        int dim = inputs[0].length;
        inputMin = new double[dim];
        inputMax = new double[dim];
        inputMean = new double[dim];
        inputStd = new double[dim];

        for (int d = 0; d < dim; d++) {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE, sum = 0;
            for (int i = 0; i < n; i++) {
                min = Math.min(min, inputs[i][d]);
                max = Math.max(max, inputs[i][d]);
                sum += inputs[i][d];
            }
            inputMin[d] = min;
            inputMax[d] = max;
            double mean = sum / n;
            inputMean[d] = mean;
            double varSum = 0;
            for (int i = 0; i < n; i++) {
                double diff = inputs[i][d] - mean;
                varSum += diff * diff;
            }
            inputStd[d] = Math.sqrt(varSum / n);
        }
    }

    public void normalize() {
        if (normalized) return;
        for (int i = 0; i < inputs.length; i++) {
            for (int d = 0; d < inputs[i].length; d++) {
                double range = inputMax[d] - inputMin[d];
                if (range > 1e-10) {
                    inputs[i][d] = (inputs[i][d] - inputMin[d]) / range;
                } else {
                    inputs[i][d] = 0;
                }
            }
        }
        normalized = true;
    }

    public void standardize() {
        if (normalized) return;
        for (int i = 0; i < inputs.length; i++) {
            for (int d = 0; d < inputs[i].length; d++) {
                if (inputStd[d] > 1e-10) {
                    inputs[i][d] = (inputs[i][d] - inputMean[d]) / inputStd[d];
                } else {
                    inputs[i][d] = 0;
                }
            }
        }
        normalized = true;
    }

    public DataSet[] split(double trainRatio) {
        int trainSize = (int) (inputs.length * trainRatio);
        int testSize = inputs.length - trainSize;

        int[] indices = new int[inputs.length];
        Random rng = new Random(42);
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        for (int i = indices.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        DataSet train = new DataSet();
        DataSet test = new DataSet();
        int inDim = inputs[0].length;
        int outDim = targets[0].length;

        train.inputs = new double[trainSize][inDim];
        train.targets = new double[trainSize][outDim];
        test.inputs = new double[testSize][inDim];
        test.targets = new double[testSize][outDim];

        for (int i = 0; i < trainSize; i++) {
            train.inputs[i] = inputs[indices[i]].clone();
            train.targets[i] = targets[indices[i]].clone();
        }
        for (int i = 0; i < testSize; i++) {
            test.inputs[i] = inputs[indices[trainSize + i]].clone();
            test.targets[i] = targets[indices[trainSize + i]].clone();
        }

        train.featureNames = featureNames;
        train.labelNames = labelNames;
        test.featureNames = featureNames;
        test.labelNames = labelNames;
        train.inputMin = inputMin; train.inputMax = inputMax;
        train.inputMean = inputMean; train.inputStd = inputStd;
        test.inputMin = inputMin; test.inputMax = inputMax;
        test.inputMean = inputMean; test.inputStd = inputStd;
        train.normalized = normalized;
        test.normalized = normalized;

        return new DataSet[]{train, test};
    }

    public String getAnalysis() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 数据集分析 ===\n");
        sb.append("来源: ").append(sourceFile).append("\n");
        sb.append("样本数: ").append(inputs.length).append("\n");
        sb.append("输入维度: ").append(inputs[0].length).append("\n");
        sb.append("输出维度: ").append(targets[0].length).append("\n");
        sb.append("归一化: ").append(normalized ? "是" : "否").append("\n\n");

        sb.append("--- 输入特征统计 ---\n");
        for (int d = 0; d < inputs[0].length; d++) {
            sb.append(String.format("  %s: min=%.4f  max=%.4f  mean=%.4f  std=%.4f\n",
                    featureNames[d], inputMin[d], inputMax[d], inputMean[d], inputStd[d]));
        }

        sb.append("\n--- 输出统计 ---\n");
        for (int d = 0; d < targets[0].length; d++) {
            double sum = 0, min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            for (int i = 0; i < targets.length; i++) {
                sum += targets[i][d];
                min = Math.min(min, targets[i][d]);
                max = Math.max(max, targets[i][d]);
            }
            String name = d < labelNames.length ? labelNames[d] : "y" + (d + 1);
            sb.append(String.format("  %s: min=%.4f  max=%.4f  mean=%.4f\n", name, min, max, sum / targets.length));
        }
        return sb.toString();
    }

    public String preview(int maxRows) {
        StringBuilder sb = new StringBuilder();
        for (String name : featureNames) sb.append(name).append(",");
        for (String name : labelNames) sb.append("[").append(name).append("],");
        sb.append("\n");

        int rows = Math.min(maxRows, inputs.length);
        for (int i = 0; i < rows; i++) {
            for (int d = 0; d < inputs[i].length; d++)
                sb.append(String.format("%.4f,", inputs[i][d]));
            for (int d = 0; d < targets[i].length; d++)
                sb.append(String.format("%.4f,", targets[i][d]));
            sb.append("\n");
        }
        if (inputs.length > maxRows) sb.append("... (共 ").append(inputs.length).append(" 行)\n");
        return sb.toString();
    }

    // Getters
    public double[][] getInputs() { return inputs; }
    public double[][] getTargets() { return targets; }
    public String[] getFeatureNames() { return featureNames; }
    public String[] getLabelNames() { return labelNames; }
    public int size() { return inputs.length; }
    public int inputDim() { return inputs[0].length; }
    public int outputDim() { return targets[0].length; }
    public boolean isNormalized() { return normalized; }
    public String getSourceFile() { return sourceFile; }
}

package machain.network;

import machain.gpu.GPUAccelerator;
import machain.gpu.GPUMatrixOps;

import java.io.*;
import java.util.Random;

public class Network implements Serializable {
    private static final long serialVersionUID = 2L;

    public enum Activation {
        SIGMOID, RELU, TANH
    }

    private int[] layerSizes;
    private Activation[] layerActivations;
    private double[][] weights;
    private double[][] biases;

    private double learningRate = 0.01;
    private double baseLearningRate = 0.01;
    private double momentum = 0.9;
    private double weightDecay = 0.0;
    private double dropoutRate = 0.0;
    private int miniBatchSize = 0;
    private String lrSchedule = "none";
    private int totalEpochs = 0;
    private int maxEpochs = 100;
    private double lastLoss = Double.MAX_VALUE;
    private double bestLoss = Double.MAX_VALUE;
    private double[][] velocityW;
    private double[][] velocityB;

    private boolean useGPU = false;
    private transient GPUMatrixOps gpuOps;

    private final Random rng = new Random(42);

    public Network(int[] layerSizes, Activation[] activations) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("至少需要2层（输入层+输出层）");
        if (activations.length != layerSizes.length - 1)
            throw new IllegalArgumentException("激活函数数量须为层数-1");

        this.layerSizes = layerSizes.clone();
        this.layerActivations = activations.clone();
        initWeights();
    }

    private void initWeights() {
        int numLayers = layerSizes.length - 1;
        weights = new double[numLayers][];
        biases = new double[numLayers][];
        velocityW = new double[numLayers][];
        velocityB = new double[numLayers][];

        for (int l = 0; l < numLayers; l++) {
            int fanIn = layerSizes[l];
            int fanOut = layerSizes[l + 1];
            double scale = Math.sqrt(2.0 / fanIn);
            if (layerActivations[l] == Activation.SIGMOID || layerActivations[l] == Activation.TANH) {
                scale = Math.sqrt(1.0 / fanIn);
            }

            weights[l] = new double[fanIn * fanOut];
            biases[l] = new double[fanOut];
            velocityW[l] = new double[fanIn * fanOut];
            velocityB[l] = new double[fanOut];

            for (int i = 0; i < weights[l].length; i++) {
                weights[l][i] = rng.nextGaussian() * scale;
            }
        }
    }

    private double activate(double x, Activation act) {
        return switch (act) {
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-clamp(x, -500, 500)));
            case RELU -> Math.max(0, x);
            case TANH -> Math.tanh(x);
        };
    }

    private double activateDerivative(double activatedOutput, Activation act) {
        return switch (act) {
            case SIGMOID -> activatedOutput * (1.0 - activatedOutput);
            case RELU -> activatedOutput > 0 ? 1.0 : 0.0;
            case TANH -> 1.0 - activatedOutput * activatedOutput;
        };
    }

    private double clamp(double x, double min, double max) {
        return Math.max(min, Math.min(max, x));
    }

    public double[] forward(double[] input) {
        if (useGPU && gpuOps != null) {
            return forwardGPU(input);
        }
        return forwardCPU(input);
    }

    private double[] forwardCPU(double[] input) {
        double[] current = input.clone();
        for (int l = 0; l < weights.length; l++) {
            int fanIn = layerSizes[l];
            int fanOut = layerSizes[l + 1];
            double[] next = new double[fanOut];
            for (int j = 0; j < fanOut; j++) {
                double sum = biases[l][j];
                for (int i = 0; i < fanIn; i++) {
                    sum += current[i] * weights[l][i * fanOut + j];
                }
                next[j] = activate(sum, layerActivations[l]);
            }
            current = next;
        }
        return current;
    }

    private double[] forwardGPU(double[] input) {
        double[] current = input.clone();
        for (int l = 0; l < weights.length; l++) {
            int fanIn = layerSizes[l];
            int fanOut = layerSizes[l + 1];

            double[][] W = new double[fanIn][fanOut];
            for (int i = 0; i < fanIn; i++) {
                for (int j = 0; j < fanOut; j++) {
                    W[i][j] = weights[l][i * fanOut + j];
                }
            }

            double[] z = gpuOps.matvec(W, current, biases[l]);

            current = new double[fanOut];
            for (int j = 0; j < fanOut; j++) {
                current[j] = activate(z[j], layerActivations[l]);
            }
        }
        return current;
    }

    public void setUseGPU(boolean useGPU) {
        this.useGPU = useGPU;
        if (useGPU && gpuOps == null) {
            GPUAccelerator acc = GPUAccelerator.getInstance();
            if (acc.isEnabled()) {
                gpuOps = acc.getMatrixOps();
            } else {
                this.useGPU = false;
            }
        }
    }

    public boolean isUsingGPU() {
        return useGPU && gpuOps != null;
    }

    public double trainBatch(double[][] inputs, double[][] targets, int epochs) {
        double totalLoss = 0;
        int n = inputs.length;
        int batchSize = (miniBatchSize > 0 && miniBatchSize < n) ? miniBatchSize : n;

        for (int epoch = 0; epoch < epochs; epoch++) {
            totalLoss = 0;
            int numLayers = weights.length;

            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            if (batchSize < n) {
                for (int i = n - 1; i > 0; i--) {
                    int j = rng.nextInt(i + 1);
                    int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
                }
            }

            for (int batchStart = 0; batchStart < n; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, n);
                int batchN = batchEnd - batchStart;

                double[][] gradW = new double[numLayers][];
                double[][] gradB = new double[numLayers][];
                for (int l = 0; l < numLayers; l++) {
                    gradW[l] = new double[weights[l].length];
                    gradB[l] = new double[biases[l].length];
                }

                for (int bi = batchStart; bi < batchEnd; bi++) {
                    int s = indices[bi];
                    double[][] layerOutputs = new double[numLayers + 1][];
                    layerOutputs[0] = inputs[s].clone();

                    for (int l = 0; l < numLayers; l++) {
                        int fanIn = layerSizes[l];
                        int fanOut = layerSizes[l + 1];
                        layerOutputs[l + 1] = new double[fanOut];
                        for (int j = 0; j < fanOut; j++) {
                            double sum = biases[l][j];
                            for (int i = 0; i < fanIn; i++) {
                                sum += layerOutputs[l][i] * weights[l][i * fanOut + j];
                            }
                            double activated = activate(sum, layerActivations[l]);
                            if (dropoutRate > 0 && l < numLayers - 1) {
                                if (rng.nextDouble() < dropoutRate) {
                                    activated = 0;
                                } else {
                                    activated /= (1.0 - dropoutRate);
                                }
                            }
                            layerOutputs[l + 1][j] = activated;
                        }
                    }

                    double[] output = layerOutputs[numLayers];
                    double sampleLoss = 0;
                    for (int k = 0; k < output.length; k++) {
                        double diff = output[k] - targets[s][k];
                        sampleLoss += diff * diff;
                    }
                    totalLoss += sampleLoss / output.length;

                    double[] delta = new double[output.length];
                    for (int k = 0; k < output.length; k++) {
                        double diff = output[k] - targets[s][k];
                        delta[k] = diff * activateDerivative(output[k], layerActivations[numLayers - 1]);
                    }

                    for (int l = numLayers - 1; l >= 0; l--) {
                        int fanIn = layerSizes[l];
                        int fanOut = layerSizes[l + 1];

                        for (int j = 0; j < fanOut; j++) {
                            gradB[l][j] += delta[j];
                        }
                        for (int i = 0; i < fanIn; i++) {
                            for (int j = 0; j < fanOut; j++) {
                                gradW[l][i * fanOut + j] += layerOutputs[l][i] * delta[j];
                            }
                        }

                        if (l > 0) {
                            double[] prevDelta = new double[fanIn];
                            for (int i = 0; i < fanIn; i++) {
                                double sum = 0;
                                for (int j = 0; j < fanOut; j++) {
                                    sum += weights[l][i * fanOut + j] * delta[j];
                                }
                                prevDelta[i] = sum * activateDerivative(layerOutputs[l][i], layerActivations[l - 1]);
                            }
                            delta = prevDelta;
                        }
                    }
                }

                for (int l = 0; l < numLayers; l++) {
                    for (int i = 0; i < weights[l].length; i++) {
                        double grad = gradW[l][i] / batchN + weightDecay * weights[l][i];
                        velocityW[l][i] = momentum * velocityW[l][i] - learningRate * grad;
                        weights[l][i] += velocityW[l][i];
                    }
                    for (int j = 0; j < biases[l].length; j++) {
                        velocityB[l][j] = momentum * velocityB[l][j] - learningRate * gradB[l][j] / batchN;
                        biases[l][j] += velocityB[l][j];
                    }
                }
            }

            totalLoss /= n;
            totalEpochs++;
            updateLearningRate();
        }

        lastLoss = totalLoss;
        if (totalLoss < bestLoss) bestLoss = totalLoss;
        return totalLoss;
    }

    private void updateLearningRate() {
        if ("cosine".equals(lrSchedule) && maxEpochs > 0) {
            double progress = (double) totalEpochs / maxEpochs;
            learningRate = baseLearningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
        } else if ("step".equals(lrSchedule)) {
            if (totalEpochs > 0 && totalEpochs % 30 == 0) {
                learningRate = baseLearningRate * Math.pow(0.5, totalEpochs / 30);
            }
        }
    }

    public double getCurrentLR() { return learningRate; }
    public double getWeightDecay() { return weightDecay; }
    public void setWeightDecay(double wd) { this.weightDecay = wd; }
    public double getDropoutRate() { return dropoutRate; }
    public void setDropoutRate(double dr) { this.dropoutRate = dr; }
    public int getMiniBatchSize() { return miniBatchSize; }
    public void setMiniBatchSize(int bs) { this.miniBatchSize = bs; }
    public String getLrSchedule() { return lrSchedule; }
    public void setLrSchedule(String schedule) { this.lrSchedule = schedule; }
    public void setMaxEpochs(int max) { this.maxEpochs = max; }
    public double getBaseLearningRate() { return baseLearningRate; }
    public void setBaseLearningRate(double lr) { this.baseLearningRate = lr; this.learningRate = lr; }

    public double evaluate(double[][] inputs, double[][] targets) {
        double totalError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] output = forward(inputs[i]);
            for (int k = 0; k < output.length; k++) {
                double diff = output[k] - targets[i][k];
                totalError += diff * diff;
            }
        }
        return totalError / (inputs.length * inputs[0].length);
    }

    public int predictClass(double[] input) {
        double[] output = forward(input);
        int best = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[best]) best = i;
        }
        return best;
    }

    public String getArchitecture() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layerSizes.length; i++) {
            sb.append(layerSizes[i]);
            if (i < layerActivations.length) {
                sb.append("-[").append(layerActivations[i]).append("]->");
            } else {
                sb.append("(output)");
            }
        }
        return sb.toString();
    }

    public int countParameters() {
        int total = 0;
        for (int l = 0; l < weights.length; l++) {
            total += weights[l].length + biases[l].length;
        }
        return total;
    }

    // Getters and setters
    public int[] getLayerSizes() { return layerSizes.clone(); }
    public Activation[] getLayerActivations() { return layerActivations.clone(); }
    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double lr) { this.learningRate = lr; }
    public double getMomentum() { return momentum; }
    public void setMomentum(double m) { this.momentum = m; }
    public int getTotalEpochs() { return totalEpochs; }
    public void setTotalEpochs(int e) { this.totalEpochs = e; }
    public double getLastLoss() { return lastLoss; }
    public void setLastLoss(double l) { this.lastLoss = l; }
    public double getBestLoss() { return bestLoss; }
    public void setBestLoss(double b) { this.bestLoss = b; }
    public double[][] getWeights() { return weights; }
    public double[][] getBiases() { return biases; }
    public String getGPUStatus() {
        return isUsingGPU() ? "GPU加速: 已启用" : "GPU加速: 未启用 (CPU模式)";
    }
}

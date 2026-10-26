package machain.network;

import java.io.*;
import java.util.*;

public class EnhancedNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum LayerType {
        DENSE, LSTM, ATTENTION
    }

    public enum Activation {
        SIGMOID, RELU, TANH
    }

    private int vocabSize;
    private int embeddingDim;
    private int hiddenDim;
    private int outputDim;
    private int sequenceLength;

    private double[][] embeddingWeights;
    private double[] embeddingBias;

    private List<LayerType> layerTypes;
    private List<int[]> layerConfigs;
    private List<double[][]> layerWeights;
    private List<double[]> layerBiases;

    private double learningRate = 0.001;
    private double dropoutRate = 0.2;
    private final Random rng = new Random(42);

    private double[][] lstmForgetW;
    private double[][] lstmInputW;
    private double[][] lstmCellW;
    private double[][] lstmOutputW;
    private double[] lstmForgetB;
    private double[] lstmInputB;
    private double[] lstmCellB;
    private double[] lstmOutputB;

    private double[] attentionW;
    private double[] attentionB;

    public EnhancedNetwork(int vocabSize, int embeddingDim, int hiddenDim, int outputDim, int sequenceLength) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.sequenceLength = sequenceLength;

        this.layerTypes = new ArrayList<>();
        this.layerConfigs = new ArrayList<>();
        this.layerWeights = new ArrayList<>();
        this.layerBiases = new ArrayList<>();

        initEmbedding();
        initLSTM();
        initAttention();
        initOutputLayer();
    }

    private void initEmbedding() {
        embeddingWeights = new double[vocabSize][embeddingDim];
        embeddingBias = new double[embeddingDim];
        double scale = Math.sqrt(2.0 / embeddingDim);
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingDim; j++) {
                embeddingWeights[i][j] = rng.nextGaussian() * scale;
            }
        }
    }

    private void initLSTM() {
        lstmForgetW = new double[embeddingDim + hiddenDim][hiddenDim];
        lstmInputW = new double[embeddingDim + hiddenDim][hiddenDim];
        lstmCellW = new double[embeddingDim + hiddenDim][hiddenDim];
        lstmOutputW = new double[embeddingDim + hiddenDim][hiddenDim];

        lstmForgetB = new double[hiddenDim];
        lstmInputB = new double[hiddenDim];
        lstmCellB = new double[hiddenDim];
        lstmOutputB = new double[hiddenDim];

        double scale = Math.sqrt(2.0 / (embeddingDim + hiddenDim));
        initMatrix(lstmForgetW, scale);
        initMatrix(lstmInputW, scale);
        initMatrix(lstmCellW, scale);
        initMatrix(lstmOutputW, scale);

        Arrays.fill(lstmForgetB, 1.0);
    }

    private void initAttention() {
        attentionW = new double[hiddenDim];
        attentionB = new double[1];
        double scale = Math.sqrt(2.0 / hiddenDim);
        for (int i = 0; i < hiddenDim; i++) {
            attentionW[i] = rng.nextGaussian() * scale;
        }
    }

    private void initOutputLayer() {
        double[][] outputW = new double[hiddenDim][outputDim];
        double[] outputB = new double[outputDim];
        double scale = Math.sqrt(2.0 / hiddenDim);
        initMatrix(outputW, scale);
        layerWeights.add(outputW);
        layerBiases.add(outputB);
    }

    private void initMatrix(double[][] matrix, double scale) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = rng.nextGaussian() * scale;
            }
        }
    }

    public double[] forward(int[] inputSequence) {
        double[][] embeddings = new double[inputSequence.length][embeddingDim];
        for (int t = 0; t < inputSequence.length; t++) {
            int wordIdx = Math.min(inputSequence[t], vocabSize - 1);
            embeddings[t] = embeddingWeights[wordIdx].clone();
        }

        double[][] lstmOutputs = forwardLSTM(embeddings);

        double[] attentionWeights = computeAttention(lstmOutputs);

        double[] context = new double[hiddenDim];
        for (int t = 0; t < lstmOutputs.length; t++) {
            for (int d = 0; d < hiddenDim; d++) {
                context[d] += attentionWeights[t] * lstmOutputs[t][d];
            }
        }

        double[] output = forwardOutput(context);
        return output;
    }

    private double[][] forwardLSTM(double[][] embeddings) {
        int seqLen = embeddings.length;
        double[][] outputs = new double[seqLen][hiddenDim];
        double[] h = new double[hiddenDim];
        double[] c = new double[hiddenDim];

        for (int t = 0; t < seqLen; t++) {
            double[] x = embeddings[t];
            double[] combined = new double[x.length + h.length];
            System.arraycopy(x, 0, combined, 0, x.length);
            System.arraycopy(h, 0, combined, x.length, h.length);

            double[] f = sigmoid(matvec(lstmForgetW, combined, lstmForgetB));
            double[] i = sigmoid(matvec(lstmInputW, combined, lstmInputB));
            double[] cCandidate = tanh(matvec(lstmCellW, combined, lstmCellB));
            double[] o = sigmoid(matvec(lstmOutputW, combined, lstmOutputB));

            for (int d = 0; d < hiddenDim; d++) {
                c[d] = f[d] * c[d] + i[d] * cCandidate[d];
                h[d] = o[d] * Math.tanh(c[d]);
            }

            outputs[t] = h.clone();
        }

        return outputs;
    }

    private double[] computeAttention(double[][] lstmOutputs) {
        double[] scores = new double[lstmOutputs.length];
        double maxScore = Double.NEGATIVE_INFINITY;

        for (int t = 0; t < lstmOutputs.length; t++) {
            double score = 0;
            for (int d = 0; d < hiddenDim; d++) {
                score += attentionW[d] * lstmOutputs[t][d];
            }
            scores[t] = score;
            maxScore = Math.max(maxScore, score);
        }

        double sumExp = 0;
        for (int t = 0; t < scores.length; t++) {
            scores[t] = Math.exp(scores[t] - maxScore);
            sumExp += scores[t];
        }

        for (int t = 0; t < scores.length; t++) {
            scores[t] /= sumExp;
        }

        return scores;
    }

    private double[] forwardOutput(double[] input) {
        double[][] W = layerWeights.get(0);
        double[] b = layerBiases.get(0);
        double[] output = new double[outputDim];

        for (int j = 0; j < outputDim; j++) {
            double sum = b[j];
            for (int i = 0; i < input.length; i++) {
                sum += input[i] * W[i][j];
            }
            output[j] = sum;
        }

        return output;
    }

    private double[] matvec(double[][] W, double[] x, double[] b) {
        int rows = W.length;
        int cols = W[0].length;
        double[] result = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = b[j];
            for (int i = 0; i < rows; i++) {
                sum += W[i][j] * x[i];
            }
            result[j] = sum;
        }
        return result;
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-clamp(x[i], -500, 500)));
        }
        return result;
    }

    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return result;
    }

    private double clamp(double x, double min, double max) {
        return Math.max(min, Math.min(max, x));
    }

    public double train(int[][] inputs, double[][] targets, int epochs) {
        double totalLoss = 0;
        int n = inputs.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            totalLoss = 0;

            for (int s = 0; s < n; s++) {
                double[] output = forward(inputs[s]);

                double sampleLoss = 0;
                for (int k = 0; k < output.length; k++) {
                    double diff = output[k] - targets[s][k];
                    sampleLoss += diff * diff;
                }
                totalLoss += sampleLoss / output.length;

                backward(inputs[s], targets[s], output);
            }

            totalLoss /= n;
        }

        return totalLoss;
    }

    private void backward(int[] inputSequence, double[] target, double[] output) {
        double[] outputError = new double[outputDim];
        for (int k = 0; k < outputDim; k++) {
            outputError[k] = output[k] - target[k];
        }

        double[][] embeddings = new double[inputSequence.length][embeddingDim];
        for (int t = 0; t < inputSequence.length; t++) {
            int wordIdx = Math.min(inputSequence[t], vocabSize - 1);
            embeddings[t] = embeddingWeights[wordIdx].clone();
        }

        double[][] lstmOutputs = forwardLSTM(embeddings);
        double[] attentionWeights = computeAttention(lstmOutputs);

        double[] context = new double[hiddenDim];
        for (int t = 0; t < lstmOutputs.length; t++) {
            for (int d = 0; d < hiddenDim; d++) {
                context[d] += attentionWeights[t] * lstmOutputs[t][d];
            }
        }

        double[][] outputW = layerWeights.get(0);
        double[] outputB = layerBiases.get(0);

        double[] contextGrad = new double[hiddenDim];
        for (int j = 0; j < outputDim; j++) {
            for (int i = 0; i < hiddenDim; i++) {
                outputW[i][j] -= learningRate * context[i] * outputError[j];
            }
            outputB[j] -= learningRate * outputError[j];
            for (int i = 0; i < hiddenDim; i++) {
                contextGrad[i] += outputW[i][j] * outputError[j];
            }
        }

        for (int t = 0; t < attentionWeights.length; t++) {
            double attnGrad = 0;
            for (int d = 0; d < hiddenDim; d++) {
                attnGrad += contextGrad[d] * lstmOutputs[t][d];
            }

            for (int d = 0; d < hiddenDim; d++) {
                attentionW[d] -= learningRate * attnGrad * (lstmOutputs[t][d] - context[d]);
            }
        }

        for (int t = 0; t < inputSequence.length; t++) {
            int wordIdx = Math.min(inputSequence[t], vocabSize - 1);
            for (int d = 0; d < embeddingDim; d++) {
                embeddingWeights[wordIdx][d] -= learningRate * 0.01 * embeddings[t][d];
            }
        }
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void setDropoutRate(double dropout) {
        this.dropoutRate = dropout;
    }

    public int getOutputDim() {
        return outputDim;
    }
}

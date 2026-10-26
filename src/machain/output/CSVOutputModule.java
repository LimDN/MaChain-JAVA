package machain.output;

import machain.network.Network;
import machain.data.DataSet;
import machain.core.TrainingResult;

import java.io.*;

public class CSVOutputModule implements OutputModule {

    @Override
    public String getName() {
        return "CSV";
    }

    @Override
    public void exportTrainingResult(TrainingResult result, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            pw.println("epoch,loss,val_loss");
            for (int i = 0; i < result.getLossHistory().size(); i++) {
                double loss = result.getLossHistory().get(i);
                double valLoss = i < result.getValLossHistory().size()
                        ? result.getValLossHistory().get(i) : Double.NaN;
                pw.printf("%d,%.8f,%.8f%n", i + 1, loss, valLoss);
            }
        }
    }

    @Override
    public void exportPredictions(Network network, DataSet dataSet, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            double[][] inputs = dataSet.getInputs();
            double[][] targets = dataSet.getTargets();

            StringBuilder header = new StringBuilder();
            String[] featureNames = dataSet.getFeatureNames();
            for (String name : featureNames) {
                header.append(name).append(",");
            }
            String[] labelNames = dataSet.getLabelNames();
            for (String name : labelNames) {
                header.append("target_").append(name).append(",");
            }
            for (String name : labelNames) {
                header.append("pred_").append(name).append(",");
            }
            header.append("error");
            pw.println(header.toString());

            for (int i = 0; i < inputs.length; i++) {
                double[] output = network.forward(inputs[i]);
                StringBuilder line = new StringBuilder();
                for (double v : inputs[i]) {
                    line.append(String.format("%.6f,", v));
                }
                for (double v : targets[i]) {
                    line.append(String.format("%.6f,", v));
                }
                for (double v : output) {
                    line.append(String.format("%.6f,", v));
                }
                double error = 0;
                for (int j = 0; j < output.length; j++) {
                    double diff = output[j] - targets[i][j];
                    error += diff * diff;
                }
                line.append(String.format("%.6f", error / output.length));
                pw.println(line.toString());
            }
        }
    }

    @Override
    public void exportModelInfo(Network network, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            pw.println("# MaChain Model Info");
            pw.println("architecture," + network.getArchitecture());
            pw.println("parameters," + network.countParameters());
            pw.println("learning_rate," + network.getLearningRate());
            pw.println("momentum," + network.getMomentum());
            pw.println("total_epochs," + network.getTotalEpochs());
            pw.println("best_loss," + network.getBestLoss());

            pw.println("\n# Layer Details");
            int[] sizes = network.getLayerSizes();
            Network.Activation[] acts = network.getLayerActivations();
            for (int i = 0; i < sizes.length; i++) {
                pw.printf("layer_%d: size=%d", i, sizes[i]);
                if (i < acts.length) {
                    pw.printf(", activation=%s", acts[i]);
                } else {
                    pw.print(", activation=LINEAR(output)");
                }
                pw.println();
            }
        }
    }
}

package machain.output;

import machain.network.Network;
import machain.data.DataSet;
import machain.core.TrainingResult;

import java.io.*;

public class JSONOutputModule implements OutputModule {

    @Override
    public String getName() {
        return "JSON";
    }

    @Override
    public void exportTrainingResult(TrainingResult result, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            pw.println("{");
            pw.println("  \"training_result\": {");
            pw.printf("    \"total_epochs\": %d,%n", result.getTotalEpochs());
            pw.printf("    \"final_loss\": %.8f,%n", result.getFinalLoss());
            pw.printf("    \"best_loss\": %.8f,%n", result.getBestLoss());
            pw.printf("    \"validation_loss\": %.8f,%n", result.getValidationLoss());
            pw.printf("    \"training_time_ms\": %d,%n", result.getTrainingTimeMs());

            pw.print("    \"loss_history\": [");
            for (int i = 0; i < result.getLossHistory().size(); i++) {
                if (i > 0) pw.print(", ");
                pw.printf("%.8f", result.getLossHistory().get(i));
            }
            pw.println("],");

            pw.print("    \"val_loss_history\": [");
            for (int i = 0; i < result.getValLossHistory().size(); i++) {
                if (i > 0) pw.print(", ");
                pw.printf("%.8f", result.getValLossHistory().get(i));
            }
            pw.println("]");

            pw.println("  }");
            pw.println("}");
        }
    }

    @Override
    public void exportPredictions(Network network, DataSet dataSet, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            double[][] inputs = dataSet.getInputs();
            double[][] targets = dataSet.getTargets();

            pw.println("{");
            pw.println("  \"predictions\": [");

            for (int i = 0; i < inputs.length; i++) {
                double[] output = network.forward(inputs[i]);
                pw.println("    {");

                pw.print("      \"input\": [");
                for (int j = 0; j < inputs[i].length; j++) {
                    if (j > 0) pw.print(", ");
                    pw.printf("%.6f", inputs[i][j]);
                }
                pw.println("],");

                pw.print("      \"target\": [");
                for (int j = 0; j < targets[i].length; j++) {
                    if (j > 0) pw.print(", ");
                    pw.printf("%.6f", targets[i][j]);
                }
                pw.println("],");

                pw.print("      \"prediction\": [");
                for (int j = 0; j < output.length; j++) {
                    if (j > 0) pw.print(", ");
                    pw.printf("%.6f", output[j]);
                }
                pw.println("],");

                double error = 0;
                for (int j = 0; j < output.length; j++) {
                    double diff = output[j] - targets[i][j];
                    error += diff * diff;
                }
                pw.printf("      \"error\": %.8f%n", error / output.length);

                pw.print("    }");
                if (i < inputs.length - 1) pw.print(",");
                pw.println();
            }

            pw.println("  ]");
            pw.println("}");
        }
    }

    @Override
    public void exportModelInfo(Network network, String outputPath) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(outputPath))) {
            pw.println("{");
            pw.println("  \"model\": {");
            pw.printf("    \"architecture\": \"%s\",%n", escapeJson(network.getArchitecture()));
            pw.printf("    \"parameters\": %d,%n", network.countParameters());
            pw.printf("    \"learning_rate\": %.6f,%n", network.getLearningRate());
            pw.printf("    \"momentum\": %.6f,%n", network.getMomentum());
            pw.printf("    \"total_epochs\": %d,%n", network.getTotalEpochs());
            pw.printf("    \"best_loss\": %.8f,%n", network.getBestLoss());

            pw.println("    \"layers\": [");
            int[] sizes = network.getLayerSizes();
            Network.Activation[] acts = network.getLayerActivations();
            for (int i = 0; i < sizes.length; i++) {
                pw.println("      {");
                pw.printf("        \"index\": %d,%n", i);
                pw.printf("        \"size\": %d,%n", sizes[i]);
                String act = i < acts.length ? acts[i].toString() : "LINEAR";
                pw.printf("        \"activation\": \"%s\"%n", act);
                pw.print("      }");
                if (i < sizes.length - 1) pw.print(",");
                pw.println();
            }
            pw.println("    ]");

            pw.println("  }");
            pw.println("}");
        }
    }

    private String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}

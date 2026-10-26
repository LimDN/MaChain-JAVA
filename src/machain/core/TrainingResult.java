package machain.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class TrainingResult implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int totalEpochs;
    private final double finalLoss;
    private final double bestLoss;
    private final double validationLoss;
    private final long trainingTimeMs;
    private final List<Double> lossHistory;
    private final List<Double> valLossHistory;
    private String note;

    public TrainingResult(int totalEpochs, double finalLoss, double bestLoss,
                          double validationLoss, long trainingTimeMs) {
        this.totalEpochs = totalEpochs;
        this.finalLoss = finalLoss;
        this.bestLoss = bestLoss;
        this.validationLoss = validationLoss;
        this.trainingTimeMs = trainingTimeMs;
        this.lossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
    }

    public void addLossPoint(double loss, double valLoss) {
        lossHistory.add(loss);
        if (!Double.isNaN(valLoss)) {
            valLossHistory.add(valLoss);
        }
    }

    public int getTotalEpochs() { return totalEpochs; }
    public double getFinalLoss() { return finalLoss; }
    public double getBestLoss() { return bestLoss; }
    public double getValidationLoss() { return validationLoss; }
    public long getTrainingTimeMs() { return trainingTimeMs; }
    public List<Double> getLossHistory() { return lossHistory; }
    public List<Double> getValLossHistory() { return valLossHistory; }
    public String getNote() { return note; }
    public void setNote(String note) { this.note = note; }

    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 训练结果 ===\n");
        sb.append(String.format("总轮数: %d\n", totalEpochs));
        sb.append(String.format("最终Loss: %.6f\n", finalLoss));
        sb.append(String.format("最优Loss: %.6f\n", bestLoss));
        if (!Double.isNaN(validationLoss)) {
            sb.append(String.format("验证Loss: %.6f\n", validationLoss));
        }
        sb.append(String.format("训练耗时: %.2f 秒\n", trainingTimeMs / 1000.0));
        return sb.toString();
    }
}

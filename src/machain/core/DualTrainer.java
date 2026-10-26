package machain.core;

import machain.data.DataSet;
import machain.network.DualStreamNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * 双流训练器 - 同时训练结构化和文本两个子网络
 */
public class DualTrainer {
    private final DualStreamNetwork dualNetwork;
    private final DataSet structTrainData;
    private final DataSet structValData;
    private final DataSet textTrainData;
    private final DataSet textValData;
    private final List<TrainingCallback> callbacks = new ArrayList<>();
    private volatile boolean stopped = false;
    private int earlyStopPatience = 0;
    private double earlyStopMinDelta = 1e-6;

    public DualTrainer(DualStreamNetwork dualNetwork,
                       DataSet structTrainData, DataSet structValData,
                       DataSet textTrainData, DataSet textValData) {
        this.dualNetwork = dualNetwork;
        this.structTrainData = structTrainData;
        this.structValData = structValData;
        this.textTrainData = textTrainData;
        this.textValData = textValData;
    }

    public void setEarlyStopping(int patience, double minDelta) {
        this.earlyStopPatience = patience;
        this.earlyStopMinDelta = minDelta;
    }

    public void addCallback(TrainingCallback callback) {
        callbacks.add(callback);
    }

    public TrainingResult train(int epochs, int logInterval) {
        stopped = false;
        long startTime = System.currentTimeMillis();

        dualNetwork.getStructNetwork().setMaxEpochs(epochs);
        dualNetwork.getTextNetwork().setMaxEpochs(epochs);

        callbacks.forEach(cb -> cb.onTrainingStart(dualNetwork.getStructNetwork(), epochs));

        double finalLoss = 0;
        double bestLoss = Double.MAX_VALUE;
        double bestValLoss = Double.MAX_VALUE;
        int patienceCounter = 0;

        for (int epoch = 1; epoch <= epochs && !stopped; epoch++) {
            double structLoss = dualNetwork.trainDual(
                    structTrainData.getInputs(), structTrainData.getTargets(),
                    textTrainData.getInputs(), textTrainData.getTargets(),
                    1
            );

            finalLoss = structLoss;

            double valLoss = Double.NaN;
            if (structValData != null && textValData != null) {
                double structVal = dualNetwork.getStructNetwork().evaluate(
                        structValData.getInputs(), structValData.getTargets());
                double textVal = dualNetwork.getTextNetwork().evaluate(
                        textValData.getInputs(), textValData.getTargets());
                valLoss = (structVal + textVal) / 2;
            }

            if (finalLoss < bestLoss) {
                bestLoss = finalLoss;
            }

            if (!Double.isNaN(valLoss) && earlyStopPatience > 0) {
                if (valLoss < bestValLoss - earlyStopMinDelta) {
                    bestValLoss = valLoss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                    if (patienceCounter >= earlyStopPatience) {
                        stopped = true;
                        final int stopEpoch = epoch;
                        final double curLoss = finalLoss;
                        final double curValLoss = valLoss;
                        callbacks.forEach(cb -> cb.onEpochEnd(stopEpoch, curLoss, curValLoss));
                        break;
                    }
                }
            }

            final double curLoss = finalLoss;
            final double curValLoss = valLoss;
            final int curEpoch = epoch;
            callbacks.forEach(cb -> cb.onEpochEnd(curEpoch, curLoss, curValLoss));
        }

        long endTime = System.currentTimeMillis();

        double finalValLoss = Double.NaN;
        if (structValData != null && textValData != null) {
            double structVal = dualNetwork.getStructNetwork().evaluate(
                    structValData.getInputs(), structValData.getTargets());
            double textVal = dualNetwork.getTextNetwork().evaluate(
                    textValData.getInputs(), textValData.getTargets());
            finalValLoss = (structVal + textVal) / 2;
        }

        TrainingResult result = new TrainingResult(
                epochs, finalLoss, bestLoss, finalValLoss, endTime - startTime
        );

        if (stopped && patienceCounter >= earlyStopPatience) {
            result.setNote("早停触发");
        }

        if (stopped && patienceCounter < earlyStopPatience) {
            callbacks.forEach(TrainingCallback::onTrainingStopped);
        } else {
            final TrainingResult finalResult = result;
            callbacks.forEach(cb -> cb.onTrainingEnd(finalResult));
        }

        return result;
    }

    public void stop() {
        stopped = true;
    }

    public boolean isStopped() { return stopped; }
    public DualStreamNetwork getDualNetwork() { return dualNetwork; }
}

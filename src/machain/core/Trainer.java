package machain.core;

import machain.network.Network;
import machain.data.DataSet;

import java.util.ArrayList;
import java.util.List;

public class Trainer {
    private final Network network;
    private final DataSet trainData;
    private final DataSet valData;
    private final List<TrainingCallback> callbacks = new ArrayList<>();
    private volatile boolean stopped = false;
    private int earlyStopPatience = 0;
    private double earlyStopMinDelta = 1e-6;

    public Trainer(Network network, DataSet trainData, DataSet valData) {
        this.network = network;
        this.trainData = trainData;
        this.valData = valData;
    }

    public void setEarlyStopping(int patience, double minDelta) {
        this.earlyStopPatience = patience;
        this.earlyStopMinDelta = minDelta;
    }

    public void addCallback(TrainingCallback callback) {
        callbacks.add(callback);
    }

    public void removeCallback(TrainingCallback callback) {
        callbacks.remove(callback);
    }

    public TrainingResult train(int epochs, int logInterval) {
        stopped = false;
        long startTime = System.currentTimeMillis();

        double[][] inputs = trainData.getInputs();
        double[][] targets = trainData.getTargets();

        network.setMaxEpochs(epochs);

        callbacks.forEach(cb -> cb.onTrainingStart(network, epochs));

        double finalLoss = 0;
        double bestLoss = network.getBestLoss();
        double bestValLoss = Double.MAX_VALUE;
        int patienceCounter = 0;
        TrainingResult result = new TrainingResult(0, 0, bestLoss, Double.NaN, 0);

        for (int epoch = 1; epoch <= epochs && !stopped; epoch++) {
            finalLoss = network.trainBatch(inputs, targets, 1);

            double valLoss = Double.NaN;
            if (valData != null) {
                valLoss = network.evaluate(valData.getInputs(), valData.getTargets());
            }

            result.addLossPoint(finalLoss, valLoss);

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

            final int curEpoch = epoch;
            final double curLoss = finalLoss;
            final double curValLoss = valLoss;
            callbacks.forEach(cb -> cb.onEpochEnd(curEpoch, curLoss, curValLoss));

            if (finalLoss < bestLoss) {
                bestLoss = finalLoss;
                network.setBestLoss(bestLoss);
            }
        }

        long endTime = System.currentTimeMillis();
        double finalValLoss = valData != null
                ? network.evaluate(valData.getInputs(), valData.getTargets())
                : Double.NaN;

        TrainingResult finalResult = new TrainingResult(
                network.getTotalEpochs(),
                finalLoss,
                bestLoss,
                finalValLoss,
                endTime - startTime
        );

        if (stopped && patienceCounter >= earlyStopPatience) {
            finalResult.setNote("早停触发于第 " + network.getTotalEpochs() + " 轮");
        }

        if (stopped && patienceCounter < earlyStopPatience) {
            callbacks.forEach(TrainingCallback::onTrainingStopped);
        } else {
            callbacks.forEach(cb -> cb.onTrainingEnd(finalResult));
        }

        return finalResult;
    }

    public void stop() {
        stopped = true;
    }

    public boolean isStopped() {
        return stopped;
    }

    public Network getNetwork() {
        return network;
    }
}

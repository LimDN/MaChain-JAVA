package machain.core;

import machain.network.Network;

public interface TrainingCallback {
    default void onTrainingStart(Network network, int totalEpochs) {}
    default void onEpochEnd(int epoch, double loss, double valLoss) {}
    default void onBatchEnd(int batch, double loss) {}
    default void onTrainingEnd(TrainingResult result) {}
    default void onTrainingStopped() {}
}

package machain.core;

import machain.network.Network;

import java.util.function.Consumer;

public class ConsoleTrainingCallback implements TrainingCallback {
    private final Consumer<String> logger;
    private final int logInterval;

    public ConsoleTrainingCallback(Consumer<String> logger, int logInterval) {
        this.logger = logger;
        this.logInterval = logInterval;
    }

    @Override
    public void onTrainingStart(Network network, int totalEpochs) {
        logger.accept("=== 训练开始 ===");
        logger.accept(String.format("网络: %s", network.getArchitecture()));
        logger.accept(String.format("参数量: %d", network.countParameters()));
        logger.accept(String.format("计划轮数: %d", totalEpochs));
    }

    @Override
    public void onEpochEnd(int epoch, double loss, double valLoss) {
        if (epoch % logInterval == 0 || epoch == 1) {
            String msg = String.format("[Epoch %d] loss=%.6f", epoch, loss);
            if (!Double.isNaN(valLoss)) {
                msg += String.format(", val_loss=%.6f", valLoss);
            }
            logger.accept(msg);
        }
    }

    @Override
    public void onTrainingEnd(TrainingResult result) {
        logger.accept("\n" + result.getSummary());
        logger.accept("=== 训练完成 ===");
    }

    @Override
    public void onTrainingStopped() {
        logger.accept("\n=== 训练被用户停止 ===");
    }
}

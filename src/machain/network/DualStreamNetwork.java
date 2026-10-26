package machain.network;

import machain.fuzzy.FuzzySkillBridge;

import java.io.Serializable;

/**
 * 双流网络 - 同时处理结构化和非结构化数据
 *
 * 两个独立的子网络（skill），通过 FuzzySkillBridge 进行交叉蒸馏
 */
public class DualStreamNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private Network structNetwork;   // 结构化数据处理网络
    private Network textNetwork;     // 文本数据处理网络
    private FuzzySkillBridge bridge; // 模糊连接器

    private double crossLossWeight = 0.1;
    private double lastCrossLoss = 0;
    private double lastStructLoss = 0;
    private double lastTextLoss = 0;
    private int totalEpochs = 0;

    public DualStreamNetwork(int[] structLayers, Network.Activation structAct,
                             int[] textLayers, Network.Activation textAct,
                             int bridgeDim) {
        Network.Activation[] structActs = new Network.Activation[structLayers.length - 1];
        for (int i = 0; i < structActs.length; i++) structActs[i] = structAct;

        Network.Activation[] textActs = new Network.Activation[textLayers.length - 1];
        for (int i = 0; i < textActs.length; i++) textActs[i] = textAct;

        this.structNetwork = new Network(structLayers, structActs);
        this.textNetwork = new Network(textLayers, textActs);

        int structHiddenDim = structLayers.length >= 2 ? structLayers[structLayers.length - 2] : structLayers[0];
        int textHiddenDim = textLayers.length >= 2 ? textLayers[textLayers.length - 2] : textLayers[0];

        this.bridge = new FuzzySkillBridge(structHiddenDim, textHiddenDim, bridgeDim);
    }

    public DualStreamNetwork(Network structNetwork, Network textNetwork, int bridgeDim) {
        this.structNetwork = structNetwork;
        this.textNetwork = textNetwork;

        int[] structSizes = structNetwork.getLayerSizes();
        int[] textSizes = textNetwork.getLayerSizes();
        int structHiddenDim = structSizes.length >= 2 ? structSizes[structSizes.length - 2] : structSizes[0];
        int textHiddenDim = textSizes.length >= 2 ? textSizes[textSizes.length - 2] : textSizes[0];

        this.bridge = new FuzzySkillBridge(structHiddenDim, textHiddenDim, bridgeDim);
    }

    /**
     * 提取结构化子网络的隐层表示
     */
    public double[] extractStructHidden(double[] structInput) {
        int[] sizes = structNetwork.getLayerSizes();
        double[] current = structInput.clone();

        Network.Activation[] acts = structNetwork.getLayerActivations();
        for (int l = 0; l < acts.length - 1; l++) {
            int fanIn = sizes[l];
            int fanOut = sizes[l + 1];
            double[] next = new double[fanOut];
            double[] weights = structNetwork.getWeights()[l];
            double[] biases = structNetwork.getBiases()[l];

            for (int j = 0; j < fanOut; j++) {
                double sum = biases[j];
                for (int i = 0; i < fanIn; i++) {
                    sum += current[i] * weights[i * fanOut + j];
                }
                next[j] = activate(sum, acts[l]);
            }
            current = next;
        }
        return current;
    }

    /**
     * 提取文本子网络的隐层表示
     */
    public double[] extractTextHidden(double[] textInput) {
        int[] sizes = textNetwork.getLayerSizes();
        double[] current = textInput.clone();

        Network.Activation[] acts = textNetwork.getLayerActivations();
        for (int l = 0; l < acts.length - 1; l++) {
            int fanIn = sizes[l];
            int fanOut = sizes[l + 1];
            double[] next = new double[fanOut];
            double[] weights = textNetwork.getWeights()[l];
            double[] biases = textNetwork.getBiases()[l];

            for (int j = 0; j < fanOut; j++) {
                double sum = biases[j];
                for (int i = 0; i < fanIn; i++) {
                    sum += current[i] * weights[i * fanOut + j];
                }
                next[j] = activate(sum, acts[l]);
            }
            current = next;
        }
        return current;
    }

    private double activate(double x, Network.Activation act) {
        return switch (act) {
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, x))));
            case RELU -> Math.max(0, x);
            case TANH -> Math.tanh(x);
        };
    }

    /**
     * 双流训练 - 同时训练两个子网络并通过桥接器进行交叉蒸馏
     */
    public double trainDual(double[][] structInputs, double[][] structTargets,
                            double[][] textInputs, double[][] textTargets,
                            int epochs) {
        double totalLoss = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // 训练结构化子网络
            double structLoss = structNetwork.trainBatch(structInputs, structTargets, 1);
            lastStructLoss = structLoss;

            // 训练文本子网络
            double textLoss = textNetwork.trainBatch(textInputs, textTargets, 1);
            lastTextLoss = textLoss;

            // 交叉蒸馏 - 用部分样本计算交叉损失并更新桥接器
            double crossLoss = 0;
            int pairs = Math.min(structInputs.length, textInputs.length);
            int sampleCount = Math.min(pairs, 8); // 每轮采样几个配对样本

            for (int s = 0; s < sampleCount; s++) {
                double[] structHidden = extractStructHidden(structInputs[s % structInputs.length]);
                double[] textHidden = extractTextHidden(textInputs[s % textInputs.length]);

                crossLoss += bridge.computeCrossLoss(structHidden, textHidden);
                bridge.updateRelation(structHidden, textHidden, structNetwork.getLearningRate());
                bridge.updateProjections(structHidden, textHidden,
                        structTargets[s % structInputs.length],
                        textTargets[s % textInputs.length],
                        structNetwork.getLearningRate());
            }

            lastCrossLoss = sampleCount > 0 ? crossLoss / sampleCount : 0;
            totalLoss = structLoss + textLoss + lastCrossLoss;
            totalEpochs++;
        }

        return totalLoss;
    }

    /**
     * 融合推理 - 同时使用两个子网络的输出
     */
    public double[] fusedPredict(double[] structInput, double[] textInput) {
        double[] structOut = structNetwork.forward(structInput);
        double[] textOut = textNetwork.forward(textInput);

        // 通过模糊隶属度加权融合
        double[] structHidden = extractStructHidden(structInput);
        double[] textHidden = extractTextHidden(textInput);
        double[] membership = bridge.computeMembership(structHidden, textHidden);

        int outDim = Math.max(structOut.length, textOut.length);
        double[] fused = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            double sVal = i < structOut.length ? structOut[i] : 0;
            double tVal = i < textOut.length ? textOut[i] : 0;
            fused[i] = membership[0] * sVal + membership[1] * tVal;
        }

        return fused;
    }

    /**
     * 单域推理 - 只用结构化子网络
     */
    public double[] predictStruct(double[] input) {
        return structNetwork.forward(input);
    }

    /**
     * 单域推理 - 只用文本子网络
     */
    public double[] predictText(double[] input) {
        return textNetwork.forward(input);
    }

    // Getters
    public Network getStructNetwork() { return structNetwork; }
    public Network getTextNetwork() { return textNetwork; }
    public FuzzySkillBridge getBridge() { return bridge; }
    public double getLastCrossLoss() { return lastCrossLoss; }
    public double getLastStructLoss() { return lastStructLoss; }
    public double getLastTextLoss() { return lastTextLoss; }
    public int getTotalEpochs() { return totalEpochs; }
    public void setTotalEpochs(int e) { this.totalEpochs = e; }

    public double getCrossLossWeight() { return crossLossWeight; }
    public void setCrossLossWeight(double w) {
        this.crossLossWeight = w;
        bridge.setCrossLossWeight(w);
    }

    public String getArchitecture() {
        return "Struct[" + structNetwork.getArchitecture() + "] + Text[" +
                textNetwork.getArchitecture() + "] + Bridge[" + bridge.getBridgeDim() + "]";
    }

    public int countParameters() {
        return structNetwork.countParameters() + textNetwork.countParameters();
    }
}

package machain.fuzzy;

import java.io.Serializable;
import java.util.Random;

/**
 * 模糊Skill连接器 - 实现交叉蒸馏的核心组件
 *
 * 功能：
 * 1. 动态隶属度函数：根据输入决定样本在多大程度上属于结构化域 vs 文本域
 * 2. 交叉蒸馏：让两个 skill 的隐空间互相渗透
 * 3. 模糊关系矩阵：学习两个域之间的关联
 */
public class FuzzySkillBridge implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int structDim;      // 结构化 skill 隐空间维度
    private final int textDim;        // 文本 skill 隐空间维度
    private final int bridgeDim;      // 桥接空间维度

    // 模糊关系矩阵：学习两个域之间的关联
    private double[][] relationMatrix;  // structDim x textDim

    // 隶属度函数参数
    private double[] membershipWeights; // 控制两个域的隶属度
    private double temperature = 1.0;   // softmax 温度，控制模糊程度

    // 交叉蒸馏投影矩阵
    private double[][] structToText;    // structDim x bridgeDim
    private double[][] textToStruct;    // textDim x bridgeDim

    // 交叉蒸馏损失权重
    private double crossLossWeight = 0.1;

    private final Random rng = new Random(42);

    public FuzzySkillBridge(int structDim, int textDim, int bridgeDim) {
        this.structDim = structDim;
        this.textDim = textDim;
        this.bridgeDim = bridgeDim;

        initMatrices();
    }

    private void initMatrices() {
        // 初始化模糊关系矩阵
        relationMatrix = new double[structDim][textDim];
        double scale = Math.sqrt(2.0 / (structDim + textDim));
        for (int i = 0; i < structDim; i++) {
            for (int j = 0; j < textDim; j++) {
                relationMatrix[i][j] = rng.nextGaussian() * scale;
            }
        }

        // 初始化隶属度权重
        membershipWeights = new double[]{0.5, 0.5}; // 初始对等

        // 初始化交叉蒸馏投影矩阵
        structToText = new double[structDim][bridgeDim];
        textToStruct = new double[textDim][bridgeDim];
        double scaleST = Math.sqrt(2.0 / structDim);
        double scaleTS = Math.sqrt(2.0 / textDim);
        for (int i = 0; i < structDim; i++) {
            for (int j = 0; j < bridgeDim; j++) {
                structToText[i][j] = rng.nextGaussian() * scaleST;
            }
        }
        for (int i = 0; i < textDim; i++) {
            for (int j = 0; j < bridgeDim; j++) {
                textToStruct[i][j] = rng.nextGaussian() * scaleTS;
            }
        }
    }

    /**
     * 计算隶属度 - 决定输入样本在多大程度上属于结构化域 vs 文本域
     * @return [mu_struct, mu_text]，和为1
     */
    public double[] computeMembership(double[] structFeatures, double[] textFeatures) {
        double structConfidence = vectorNorm(structFeatures);
        double textConfidence = vectorNorm(textFeatures);

        // 通过 softmax 计算隶属度，温度控制模糊程度
        double expStruct = Math.exp(structConfidence / temperature);
        double expText = Math.exp(textConfidence / temperature);
        double sum = expStruct + expText;

        return new double[]{expStruct / sum, expText / sum};
    }

    /**
     * 交叉蒸馏 - 将结构化 skill 的表示投影到文本空间
     */
    public double[] projectStructToText(double[] structHidden) {
        double[] projected = new double[bridgeDim];
        for (int j = 0; j < bridgeDim; j++) {
            double sum = 0;
            for (int i = 0; i < structDim; i++) {
                sum += structHidden[i] * structToText[i][j];
            }
            projected[j] = sigmoid(sum);
        }
        return projected;
    }

    /**
     * 交叉蒸馏 - 将文本 skill 的表示投影到结构化空间
     */
    public double[] projectTextToStruct(double[] textHidden) {
        double[] projected = new double[bridgeDim];
        for (int j = 0; j < bridgeDim; j++) {
            double sum = 0;
            for (int i = 0; i < textDim; i++) {
                sum += textHidden[i] * textToStruct[i][j];
            }
            projected[j] = sigmoid(sum);
        }
        return projected;
    }

    /**
     * 计算交叉蒸馏损失 - 衡量两个 skill 隐空间的对齐程度
     */
    public double computeCrossLoss(double[] structHidden, double[] textHidden) {
        double[] structProj = projectStructToText(structHidden);
        double[] textProj = projectTextToStruct(textHidden);

        // 两个投影应该在桥接空间中接近
        double loss = 0;
        int dim = Math.min(structProj.length, textProj.length);
        for (int i = 0; i < dim; i++) {
            double diff = structProj[i] - textProj[i];
            loss += diff * diff;
        }

        return crossLossWeight * loss / dim;
    }

    /**
     * 更新模糊关系矩阵 - 基于梯度下降
     */
    public void updateRelation(double[] structHidden, double[] textHidden, double lr) {
        // 计算关系矩阵的梯度（简化的 Hebbian 学习）
        for (int i = 0; i < structDim; i++) {
            for (int j = 0; j < textDim; j++) {
                double gradient = structHidden[i] * textHidden[j];
                relationMatrix[i][j] -= lr * gradient;
                // 权重衰减
                relationMatrix[i][j] *= 0.999;
            }
        }
    }

    /**
     * 更新投影矩阵
     */
    public void updateProjections(double[] structHidden, double[] textHidden,
                                   double[] structTarget, double[] textTarget, double lr) {
        // 更新 structToText
        double[] structProj = projectStructToText(structHidden);
        for (int i = 0; i < structDim; i++) {
            for (int j = 0; j < bridgeDim; j++) {
                double error = (j < textTarget.length ? textTarget[j] : 0) - structProj[j];
                double grad = error * structHidden[i] * structProj[j] * (1 - structProj[j]);
                structToText[i][j] += lr * grad;
            }
        }

        // 更新 textToStruct
        double[] textProj = projectTextToStruct(textHidden);
        for (int i = 0; i < textDim; i++) {
            for (int j = 0; j < bridgeDim; j++) {
                double error = (j < structTarget.length ? structTarget[j] : 0) - textProj[j];
                double grad = error * textHidden[i] * textProj[j] * (1 - textProj[j]);
                textToStruct[i][j] += lr * grad;
            }
        }
    }

    /**
     * 通过模糊关系矩阵进行推理
     */
    public double[] fuzzyInference(double[] structInput, double[] textInput) {
        double[] membership = computeMembership(structInput, textInput);
        double muStruct = membership[0];
        double muText = membership[1];

        // 加权融合两个域的表示
        double[] structProj = projectStructToText(structInput);
        double[] textProj = projectTextToStruct(textInput);

        double[] result = new double[bridgeDim];
        for (int i = 0; i < bridgeDim; i++) {
            result[i] = muStruct * (i < structProj.length ? structProj[i] : 0)
                      + muText * (i < textProj.length ? textProj[i] : 0);
        }

        return result;
    }

    private double vectorNorm(double[] v) {
        double sum = 0;
        for (double x : v) sum += x * x;
        return Math.sqrt(sum);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-clamp(x, -500, 500)));
    }

    private double clamp(double x, double min, double max) {
        return Math.max(min, Math.min(max, x));
    }

    // Getters and setters
    public double getCrossLossWeight() { return crossLossWeight; }
    public void setCrossLossWeight(double w) { this.crossLossWeight = w; }
    public double getTemperature() { return temperature; }
    public void setTemperature(double t) { this.temperature = Math.max(0.01, t); }
    public int getStructDim() { return structDim; }
    public int getTextDim() { return textDim; }
    public int getBridgeDim() { return bridgeDim; }
    public double[][] getRelationMatrix() { return relationMatrix; }
    public double[] getMembershipWeights() { return membershipWeights.clone(); }
}

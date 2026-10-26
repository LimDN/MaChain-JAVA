package machain.network;

public enum OutputMode {
    REGRESSION("回归", "连续值输出"),
    BINARY_CLASSIFICATION("二分类", "Sigmoid输出概率"),
    MULTICLASS_CLASSIFICATION("多分类", "Softmax输出概率分布"),
    MULTILABEL("多标签", "多个Sigmoid独立输出"),
    PROBABILITY_DISTRIBUTION("概率分布", "归一化概率输出"),
    AUTONOMOUS_GENERATION("自主生成", "模型自主决定输出文本");

    private final String displayName;
    private final String description;

    OutputMode(String displayName, String description) {
        this.displayName = displayName;
        this.description = description;
    }

    public String getDisplayName() { return displayName; }
    public String getDescription() { return description; }

    public double[] activateOutput(double[] rawOutput) {
        return switch (this) {
            case REGRESSION -> rawOutput;
            case AUTONOMOUS_GENERATION -> rawOutput;
            case BINARY_CLASSIFICATION -> {
                double[] result = new double[rawOutput.length];
                for (int i = 0; i < rawOutput.length; i++) {
                    result[i] = 1.0 / (1.0 + Math.exp(-rawOutput[i]));
                }
                yield result;
            }
            case MULTICLASS_CLASSIFICATION, PROBABILITY_DISTRIBUTION -> softmax(rawOutput);
            case MULTILABEL -> {
                double[] result = new double[rawOutput.length];
                for (int i = 0; i < rawOutput.length; i++) {
                    result[i] = 1.0 / (1.0 + Math.exp(-rawOutput[i]));
                }
                yield result;
            }
        };
    }

    public int predictClass(double[] output) {
        double[] activated = activateOutput(output);
        int best = 0;
        for (int i = 1; i < activated.length; i++) {
            if (activated[i] > activated[best]) best = i;
        }
        return best;
    }

    private double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) max = Math.max(max, v);
        double[] exp = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
}

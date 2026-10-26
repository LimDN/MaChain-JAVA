package machain.generation;

import machain.data.TextProcessor;
import machain.network.Network;
import machain.network.EnhancedNetwork;

import java.util.*;

/**
 * 模型自主生成器 — 结合神经网络隐层表示与统计语言模型进行文本生成。
 *
 * 原理：
 * 1. 从训练文本构建字符/词级 n-gram 转移概率表
 * 2. 利用已训练网络的隐层激活作为"风格种子"
 * 3. 隐层激活经 softmax 映射为词汇空间上的偏置分布
 * 4. 自回归逐 token 生成：n-gram 概率 × 网络偏置 → 采样下一个 token
 */
public class TextGenerator {

    private final Map<String, Map<String, Integer>> bigramCounts = new HashMap<>();
    private final Map<String, Map<String, Integer>> trigramCounts = new HashMap<>();
    private final Map<String, Integer> unigramCounts = new HashMap<>();
    private List<String> vocabulary = new ArrayList<>();
    private boolean wordLevel = true;
    private final Random rng = new Random();

    private double temperature = 1.0;
    private double networkBiasWeight = 0.3;
    private int maxGenLength = 200;
    private double repetitionPenalty = 0.5;

    public TextProcessor textProcessor;
    public Network network;
    public EnhancedNetwork enhancedNetwork;

    public TextGenerator() {}

    public void setTemperature(double t) { this.temperature = Math.max(0.01, t); }
    public double getTemperature() { return temperature; }
    public void setNetworkBiasWeight(double w) { this.networkBiasWeight = Math.max(0, Math.min(1, w)); }
    public double getNetworkBiasWeight() { return networkBiasWeight; }
    public void setMaxGenLength(int len) { this.maxGenLength = Math.max(10, len); }

    public void buildFromTexts(List<String> texts, boolean wordLevelMode) {
        this.wordLevel = wordLevelMode;
        bigramCounts.clear();
        trigramCounts.clear();
        unigramCounts.clear();
        vocabulary.clear();
        Set<String> vocabSet = new LinkedHashSet<>();

        for (String text : texts) {
            List<String> tokens = tokenize(text);
            for (String t : tokens) vocabSet.add(t);

            for (int i = 0; i < tokens.size(); i++) {
                String tok = tokens.get(i);
                unigramCounts.merge(tok, 1, Integer::sum);

                if (i > 0) {
                    String prev = tokens.get(i - 1);
                    bigramCounts.computeIfAbsent(prev, k -> new HashMap<>())
                            .merge(tok, 1, Integer::sum);
                }
                if (i > 1) {
                    String pp = tokens.get(i - 2);
                    String p = tokens.get(i - 1);
                    String key = pp + " " + p;
                    trigramCounts.computeIfAbsent(key, k -> new HashMap<>())
                            .merge(tok, 1, Integer::sum);
                }
            }
        }

        vocabulary.addAll(vocabSet);
    }

    public int getVocabSize() { return vocabulary.size(); }
    public int getCorpusTokenCount() { return unigramCounts.values().stream().mapToInt(Integer::intValue).sum(); }

    public String generate(String seed) {
        if (vocabulary.isEmpty()) return "[错误: 未构建语言模型]";

        List<String> context = new ArrayList<>();
        if (seed != null && !seed.trim().isEmpty()) {
            context.addAll(tokenize(seed));
        }
        if (context.isEmpty()) {
            context.add(pickWeightedUnigram());
        }

        double[] networkBias = computeNetworkBias(context);

        StringBuilder generated = new StringBuilder();
        for (String tok : context) {
            if (generated.length() > 0) {
                if (wordLevel && !isPunctuation(tok)) {
                    generated.append(" ");
                }
            }
            generated.append(tok);
        }

        for (int step = 0; step < maxGenLength; step++) {
            String next = sampleNextToken(context, networkBias);
            if (next == null || next.equals("<END>")) break;

            if (wordLevel) {
                if (!isPunctuation(next)) {
                    generated.append(" ");
                }
                generated.append(next);
            } else {
                generated.append(next);
            }

            context.add(next);
            if (context.size() > 3) context.remove(0);
        }

        return generated.toString().trim();
    }

    private boolean isPunctuation(String token) {
        if (token == null || token.length() != 1) return false;
        char c = token.charAt(0);
        return Character.getType(c) == Character.CONNECTOR_PUNCTUATION
            || Character.getType(c) == Character.DASH_PUNCTUATION
            || Character.getType(c) == Character.START_PUNCTUATION
            || Character.getType(c) == Character.END_PUNCTUATION
            || Character.getType(c) == Character.INITIAL_QUOTE_PUNCTUATION
            || Character.getType(c) == Character.FINAL_QUOTE_PUNCTUATION
            || Character.getType(c) == Character.OTHER_PUNCTUATION;
    }

    public List<String> generateMultiple(String seed, int count) {
        List<String> results = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            results.add(generate(seed));
        }
        return results;
    }

    private String sampleNextToken(List<String> context, double[] networkBias) {
        Map<String, Double> candidates = new HashMap<>();

        if (context.size() >= 2) {
            String key = context.get(context.size() - 2) + " " + context.get(context.size() - 1);
            Map<String, Integer> tri = trigramCounts.get(key);
            if (tri != null) {
                int total = tri.values().stream().mapToInt(Integer::intValue).sum();
                for (Map.Entry<String, Integer> e : tri.entrySet()) {
                    candidates.merge(e.getKey(), (double) e.getValue() / total, Double::sum);
                }
            }
        }

        if (!context.isEmpty()) {
            String prev = context.get(context.size() - 1);
            Map<String, Integer> bi = bigramCounts.get(prev);
            if (bi != null) {
                int total = bi.values().stream().mapToInt(Integer::intValue).sum();
                for (Map.Entry<String, Integer> e : bi.entrySet()) {
                    double prob = (double) e.getValue() / total;
                    candidates.merge(e.getKey(), prob * 0.7, Double::sum);
                }
            }
        }

        if (candidates.isEmpty()) {
            for (String tok : unigramCounts.keySet()) {
                int count = unigramCounts.get(tok);
                double prob = (double) count / getCorpusTokenCount();
                candidates.put(tok, prob);
            }
        }

        if (networkBias != null && networkBias.length > 0) {
            for (int i = 0; i < Math.min(vocabulary.size(), networkBias.length); i++) {
                String tok = vocabulary.get(i);
                double bias = networkBias[i] * networkBiasWeight;
                candidates.merge(tok, Math.max(0, bias), Double::sum);
            }
        }

        for (String recent : context) {
            if (candidates.containsKey(recent)) {
                candidates.put(recent, candidates.get(recent) * repetitionPenalty);
            }
        }

        return sampleFromCandidates(candidates);
    }

    private String sampleFromCandidates(Map<String, Double> candidates) {
        if (candidates.isEmpty()) return null;

        double maxScore = candidates.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        if (maxScore <= 0) maxScore = 1.0;

        double totalWeight = 0;
        Map<String, Double> weighted = new HashMap<>();
        for (Map.Entry<String, Double> e : candidates.entrySet()) {
            double w = Math.pow(e.getValue() / maxScore, 1.0 / temperature);
            weighted.put(e.getKey(), w);
            totalWeight += w;
        }

        if (totalWeight <= 0) {
            List<String> keys = new ArrayList<>(candidates.keySet());
            return keys.get(rng.nextInt(keys.size()));
        }

        double r = rng.nextDouble() * totalWeight;
        double cumulative = 0;
        for (Map.Entry<String, Double> e : weighted.entrySet()) {
            cumulative += e.getValue();
            if (r <= cumulative) return e.getKey();
        }

        return weighted.keySet().iterator().next();
    }

    private double[] computeNetworkBias(List<String> context) {
        if (vocabulary.isEmpty()) return null;

        try {
            if (enhancedNetwork != null && textProcessor != null) {
                String contextText = String.join(wordLevel ? " " : "", context);
                int[] sequence = textProcessor.textToSequence(contextText, 50);
                double[] output = enhancedNetwork.forward(sequence);
                
                double[] bias = new double[vocabulary.size()];
                for (int v = 0; v < vocabulary.size(); v++) {
                    double score = 0;
                    for (int o = 0; o < output.length; o++) {
                        score += output[o] * Math.sin((v + 1) * (o + 1) * 0.1);
                    }
                    bias[v] = 1.0 / (1.0 + Math.exp(-score));
                }
                
                double sum = 0;
                for (double b : bias) sum += b;
                if (sum > 0) {
                    for (int i = 0; i < bias.length; i++) bias[i] /= sum;
                }
                
                return bias;
            }
            
            if (network == null || textProcessor == null) return null;

            String contextText = String.join(wordLevel ? " " : "", context);
            double[] input = textProcessor.vectorize(contextText);
            int inputDim = network.getLayerSizes()[0];
            if (input.length != inputDim) {
                double[] padded = new double[inputDim];
                System.arraycopy(input, 0, padded, 0, Math.min(input.length, inputDim));
                input = padded;
            }

            double[] hidden = extractHidden(input);

            double[] bias = new double[vocabulary.size()];
            for (int v = 0; v < vocabulary.size(); v++) {
                double score = 0;
                for (int h = 0; h < hidden.length; h++) {
                    score += hidden[h % hidden.length] * Math.sin((v + 1) * (h + 1) * 0.1);
                }
                bias[v] = 1.0 / (1.0 + Math.exp(-score));
            }

            double sum = 0;
            for (double b : bias) sum += b;
            if (sum > 0) {
                for (int i = 0; i < bias.length; i++) bias[i] /= sum;
            }

            return bias;
        } catch (Exception e) {
            return null;
        }
    }

    private double[] extractHidden(double[] input) {
        int[] sizes = network.getLayerSizes();
        Network.Activation[] acts = network.getLayerActivations();
        double[][] weights = network.getWeights();
        double[][] biases = network.getBiases();

        double[] current = input.clone();
        int lastHiddenLayer = acts.length - 1;

        for (int l = 0; l < lastHiddenLayer; l++) {
            int fanIn = sizes[l];
            int fanOut = sizes[l + 1];
            double[] next = new double[fanOut];
            for (int j = 0; j < fanOut; j++) {
                double sum = biases[l][j];
                for (int i = 0; i < fanIn; i++) {
                    sum += current[i] * weights[l][i * fanOut + j];
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

    private String pickWeightedUnigram() {
        int total = getCorpusTokenCount();
        int r = rng.nextInt(total);
        int cumulative = 0;
        for (Map.Entry<String, Integer> e : unigramCounts.entrySet()) {
            cumulative += e.getValue();
            if (r < cumulative) return e.getKey();
        }
        return vocabulary.get(0);
    }

    private boolean preservePunctuation = false;

    public void setPreservePunctuation(boolean preserve) {
        this.preservePunctuation = preserve;
    }

    public boolean isPreservePunctuation() {
        return preservePunctuation;
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (wordLevel) {
            if (preservePunctuation) {
                String cleaned = text.replaceAll("[^\\p{L}\\p{N}\\p{P}\\s]", " ").trim();
                StringBuilder current = new StringBuilder();
                for (char c : cleaned.toCharArray()) {
                    if (Character.isLetterOrDigit(c)) {
                        current.append(c);
                    } else if (Character.getType(c) == Character.CONNECTOR_PUNCTUATION
                            || Character.getType(c) == Character.DASH_PUNCTUATION
                            || Character.getType(c) == Character.START_PUNCTUATION
                            || Character.getType(c) == Character.END_PUNCTUATION
                            || Character.getType(c) == Character.INITIAL_QUOTE_PUNCTUATION
                            || Character.getType(c) == Character.FINAL_QUOTE_PUNCTUATION
                            || Character.getType(c) == Character.OTHER_PUNCTUATION) {
                        if (current.length() > 0) {
                            tokens.add(current.toString());
                            current = new StringBuilder();
                        }
                        tokens.add(String.valueOf(c));
                    } else if (Character.isWhitespace(c)) {
                        if (current.length() > 0) {
                            tokens.add(current.toString());
                            current = new StringBuilder();
                        }
                    }
                }
                if (current.length() > 0) {
                    tokens.add(current.toString());
                }
            } else {
                String cleaned = text.replaceAll("[^\\p{L}\\p{N}\\s]", " ").trim();
                for (String w : cleaned.split("\\s+")) {
                    if (!w.isEmpty()) tokens.add(w);
                }
            }
        } else {
            for (char c : text.toCharArray()) {
                if (!Character.isWhitespace(c)) {
                    tokens.add(String.valueOf(c));
                }
            }
        }
        return tokens;
    }

    public String getStats() {
        return String.format("词表: %d | 语料token: %d | bigram: %d | trigram: %d | 温度: %.2f | 网络偏置权重: %.2f",
                vocabulary.size(), getCorpusTokenCount(),
                bigramCounts.size(), trigramCounts.size(),
                temperature, networkBiasWeight);
    }
}

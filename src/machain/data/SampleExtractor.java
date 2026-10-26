package machain.data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

public class SampleExtractor {

    public enum Strategy {
        ARTICLE_TOPIC("文章段落主题"),
        BILINGUAL_PAIR("双语对照"),
        SEQUENCE_PREDICT("序列预测"),
        SENTIMENT("情感分析"),
        SUMMARY_EXTRACT("摘要提取"),
        KEYWORD_PREDICT("关键词预测"),
        AUGMENT("数据增强");

        public final String label;
        Strategy(String label) { this.label = label; }

        public String description() {
            switch (this) {
                case ARTICLE_TOPIC: return "从文章段落自动提取主题分类样本";
                case BILINGUAL_PAIR: return "从双语对照文本提取翻译对";
                case SEQUENCE_PREDICT: return "用上下文预测下一句/段";
                case SENTIMENT: return "基于关键词词典自动标注情感";
                case SUMMARY_EXTRACT: return "从长短段落对提取摘要样本";
                case KEYWORD_PREDICT: return "从文本提取关键词预测样本";
                case AUGMENT: return "对已有文本做同义替换/扰动生成新样本";
                default: return "";
            }
        }
    }

    public enum ChunkingStrategy {
        FIXED("固定分块", "使用固定大小的上下文窗口"),
        LENGTH_BASED("长度自适应", "根据句子长度和段落结构调整块大小"),
        COMPLEXITY_BASED("复杂度自适应", "根据句子复杂度（词汇密度、从句数量）调整"),
        ENTROPY_BASED("信息熵分块", "在信息密度突变处切分");

        public final String label;
        public final String description;

        ChunkingStrategy(String label, String description) {
            this.label = label;
            this.description = description;
        }

        public static ChunkingStrategy fromLabel(String label) {
            for (ChunkingStrategy cs : values()) {
                if (cs.label.equals(label)) return cs;
            }
            return FIXED;
        }
    }

    public static class ExtractedSample {
        public final String inputText;
        public final String targetLabel;

        public ExtractedSample(String inputText, String targetLabel) {
            this.inputText = inputText;
            this.targetLabel = targetLabel;
        }
    }

    private Strategy strategy;
    private ChunkingStrategy chunkingStrategy;
    private Map<String, String> params;
    private Random rng;

    private static final String[] POSITIVE_WORDS = {
        "好", "优秀", "出色", "棒", "赞", "喜欢", "成功", "进步", "快乐", "幸福",
        "满意", "精彩", "美丽", "温暖", "感谢", "希望", "强大", "完美", "舒适", "方便",
        "good", "great", "excellent", "amazing", "wonderful", "happy", "love", "best",
        "beautiful", "fantastic", "awesome", "perfect", "brilliant", "superb", "nice"
    };

    private static final String[] NEGATIVE_WORDS = {
        "差", "糟糕", "失败", "糟糕", "讨厌", "痛苦", "失望", "悲伤", "愤怒", "恐惧",
        "问题", "困难", "危险", "错误", "缺陷", "糟糕", "恶心", "无聊", "烦人", "混乱",
        "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly", "poor",
        "disgusting", "boring", "annoying", "dangerous", "failure", "wrong", "sad"
    };

    private static final Map<String, String[]> SYNONYM_MAP = new LinkedHashMap<>();
    static {
        SYNONYM_MAP.put("好", new String[]{"优", "佳", "良"});
        SYNONYM_MAP.put("大", new String[]{"巨", "硕", "宏"});
        SYNONYM_MAP.put("小", new String[]{"微", "细", "窄"});
        SYNONYM_MAP.put("快", new String[]{"速", "疾", "迅"});
        SYNONYM_MAP.put("好", new String[]{"优", "佳", "妙"});
        SYNONYM_MAP.put("漂亮", new String[]{"美丽", "好看", "标致"});
        SYNONYM_MAP.put("高兴", new String[]{"开心", "快乐", "愉悦"});
        SYNONYM_MAP.put("困难", new String[]{"艰难", "费劲", "棘手"});
        SYNONYM_MAP.put("good", new String[]{"great", "fine", "nice"});
        SYNONYM_MAP.put("big", new String[]{"large", "huge", "great"});
        SYNONYM_MAP.put("small", new String[]{"little", "tiny", "minor"});
        SYNONYM_MAP.put("fast", new String[]{"quick", "rapid", "swift"});
        SYNONYM_MAP.put("happy", new String[]{"glad", "joyful", "cheerful"});
        SYNONYM_MAP.put("bad", new String[]{"poor", "terrible", "awful"});
    }

    public SampleExtractor(Strategy strategy) {
        this.strategy = strategy;
        this.chunkingStrategy = ChunkingStrategy.FIXED;
        this.params = new HashMap<>();
        this.rng = new Random(42);
    }

    public void setChunkingStrategy(ChunkingStrategy strategy) {
        this.chunkingStrategy = strategy;
    }

    public ChunkingStrategy getChunkingStrategy() {
        return chunkingStrategy;
    }

    public static String[] chunkingStrategyLabels() {
        ChunkingStrategy[] values = ChunkingStrategy.values();
        String[] labels = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            labels[i] = values[i].label;
        }
        return labels;
    }

    public void setParam(String key, String value) {
        params.put(key, value);
    }

    public String getParam(String key, String defaultValue) {
        return params.getOrDefault(key, defaultValue);
    }

    public int getParamInt(String key, int defaultValue) {
        String v = params.get(key);
        if (v == null) return defaultValue;
        try { return Integer.parseInt(v); } catch (Exception e) { return defaultValue; }
    }

    public List<ExtractedSample> extract(List<String> rawTexts) {
        switch (strategy) {
            case ARTICLE_TOPIC: return extractArticleTopic(rawTexts);
            case BILINGUAL_PAIR: return extractBilingualPair(rawTexts);
            case SEQUENCE_PREDICT: return extractSequencePredict(rawTexts);
            case SENTIMENT: return extractSentiment(rawTexts);
            case SUMMARY_EXTRACT: return extractSummary(rawTexts);
            case KEYWORD_PREDICT: return extractKeywordPredict(rawTexts);
            case AUGMENT: return extractAugment(rawTexts);
            default: return Collections.emptyList();
        }
    }

    public List<ExtractedSample> extractFromFile(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
        return extractFromLines(lines);
    }

    public List<ExtractedSample> extractFromLines(List<String> lines) {
        switch (strategy) {
            case BILINGUAL_PAIR:
                return extractBilingualFromLines(lines);
            case ARTICLE_TOPIC:
                return extractArticleFromLines(lines);
            default:
                String fullText = String.join("\n", lines);
                return extract(Collections.singletonList(fullText));
        }
    }

    public DataSet toDataSet(List<ExtractedSample> samples, TextProcessor processor) {
        if (samples.isEmpty()) return new DataSet();

        List<String> inputTexts = samples.stream().map(s -> s.inputText).collect(Collectors.toList());
        processor.buildVocabulary(inputTexts);
        int vectorSize = processor.getVectorSize();

        Map<String, Integer> labelMap = new LinkedHashMap<>();
        for (ExtractedSample s : samples) {
            if (!labelMap.containsKey(s.targetLabel)) {
                labelMap.put(s.targetLabel, labelMap.size());
            }
        }
        int numLabels = labelMap.size();

        double[][] inputs = new double[samples.size()][vectorSize];
        double[][] targets = new double[samples.size()][numLabels];

        for (int i = 0; i < samples.size(); i++) {
            inputs[i] = processor.vectorize(samples.get(i).inputText);
            int labelIdx = labelMap.get(samples.get(i).targetLabel);
            targets[i][labelIdx] = 1.0;
        }

        String[] featNames = new String[vectorSize];
        for (int i = 0; i < vectorSize; i++) featNames[i] = "feat_" + i;

        return DataSet.fromArrays(inputs, targets, featNames,
                labelMap.keySet().toArray(new String[0]));
    }

    // ========== 文章段落主题 ==========

    private List<ExtractedSample> extractArticleTopic(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();
        for (String text : rawTexts) {
            List<String> paragraphs = splitParagraphs(text);
            if (paragraphs.size() < 2) continue;

            for (int i = 0; i < paragraphs.size(); i++) {
                String p = paragraphs.get(i).trim();
                if (p.length() < 5) continue;

                String section;
                double ratio = (double) i / Math.max(1, paragraphs.size() - 1);
                if (ratio < 0.25) section = "开头";
                else if (ratio < 0.75) section = "中段";
                else section = "结尾";

                samples.add(new ExtractedSample(p, section));
            }
        }
        return samples;
    }

    private List<ExtractedSample> extractArticleFromLines(List<String> lines) {
        List<ExtractedSample> samples = new ArrayList<>();
        List<String> currentSection = new ArrayList<>();
        String currentHeader = "未分类";

        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) continue;

            if (isHeaderLine(trimmed)) {
                if (!currentSection.isEmpty()) {
                    String content = String.join(" ", currentSection);
                    if (content.length() >= 5) {
                        samples.add(new ExtractedSample(content, currentHeader));
                    }
                    currentSection.clear();
                }
                currentHeader = trimmed.replaceAll("[#\\*\\d\\.\\s]+", "").trim();
                if (currentHeader.isEmpty()) currentHeader = "段落";
            } else {
                currentSection.add(trimmed);
            }
        }
        if (!currentSection.isEmpty()) {
            String content = String.join(" ", currentSection);
            if (content.length() >= 5) {
                samples.add(new ExtractedSample(content, currentHeader));
            }
        }

        if (samples.size() <= 1) {
            samples.addAll(extractArticleTopic(Collections.singletonList(String.join("\n", lines))));
        }

        return samples;
    }

    private boolean isHeaderLine(String line) {
        if (line.matches("^#+\\s+.*")) return true;
        if (line.matches("^\\d+[\\.\\、]\\s*\\S+")) return true;
        if (line.matches("^[\\*\\-]\\s+\\S+")) return true;
        if (line.length() <= 40 && !line.contains("，") && !line.contains("。")
                && !line.contains(",") && !line.contains(".")) return true;
        return false;
    }

    // ========== 双语对照 ==========

    private List<ExtractedSample> extractBilingualPair(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();
        String direction = getParam("direction", "en2zh");

        for (String text : rawTexts) {
            List<String[]> pairs = parseBilingualLines(text);
            for (String[] pair : pairs) {
                if (pair[0].length() < 2 || pair[1].length() < 2) continue;
                if ("en2zh".equals(direction)) {
                    samples.add(new ExtractedSample(pair[0], pair[1]));
                } else {
                    samples.add(new ExtractedSample(pair[1], pair[0]));
                }
            }
        }
        return samples;
    }

    private List<ExtractedSample> extractBilingualFromLines(List<String> lines) {
        List<ExtractedSample> samples = new ArrayList<>();
        String direction = getParam("direction", "en2zh");
        String separator = getParam("separator", "auto");

        for (int i = 0; i + 1 < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;

            String[] parts = splitBilingualLine(line, separator);
            if (parts != null && parts[0].length() >= 2 && parts[1].length() >= 2) {
                if ("en2zh".equals(direction)) {
                    samples.add(new ExtractedSample(parts[0], parts[1]));
                } else {
                    samples.add(new ExtractedSample(parts[1], parts[0]));
                }
                continue;
            }

            String nextLine = lines.get(i + 1).trim();
            if (nextLine.isEmpty()) continue;

            boolean firstIsEn = containsEnglish(line) && !containsChinese(line);
            boolean secondIsZh = containsChinese(nextLine) && !containsEnglish(nextLine);
            boolean firstIsZh = containsChinese(line) && !containsEnglish(line);
            boolean secondIsEn = containsEnglish(nextLine) && !containsChinese(nextLine);

            if (firstIsEn && secondIsZh) {
                samples.add(new ExtractedSample(line, nextLine));
                i++;
            } else if (firstIsZh && secondIsEn) {
                if ("en2zh".equals(direction)) {
                    samples.add(new ExtractedSample(nextLine, line));
                } else {
                    samples.add(new ExtractedSample(line, nextLine));
                }
                i++;
            }
        }
        return samples;
    }

    private String[] splitBilingualLine(String line, String separator) {
        String[] seps;
        if ("auto".equals(separator)) {
            seps = new String[]{"\\t", "\\s*\\|\\|\\|\\s*", "\\s*══+\\s*", "\\s*━━+\\s*",
                    "\\s*→\\s*", "\\s*<->\\s*", "\\s*===\\s*"};
        } else {
            seps = new String[]{Pattern.quote(separator)};
        }

        for (String sep : seps) {
            String[] parts = line.split(sep, 2);
            if (parts.length == 2 && parts[0].trim().length() >= 2 && parts[1].trim().length() >= 2) {
                return new String[]{parts[0].trim(), parts[1].trim()};
            }
        }
        return null;
    }

    private List<String[]> parseBilingualLines(String text) {
        List<String[]> pairs = new ArrayList<>();
        String[] lines = text.split("\\n");
        String separator = getParam("separator", "auto");

        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty()) continue;

            String[] parts = splitBilingualLine(line, separator);
            if (parts != null) {
                pairs.add(parts);
            }
        }
        return pairs;
    }

    // ========== 序列预测 ==========

    private List<ExtractedSample> extractSequencePredict(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();
        int baseContextSize = getParamInt("contextSize", 2);

        for (String text : rawTexts) {
            List<String> sentences = splitSentences(text);
            if (sentences.size() < 2) continue;

            switch (chunkingStrategy) {
                case FIXED:
                    samples.addAll(extractFixedContext(sentences, baseContextSize));
                    break;
                case LENGTH_BASED:
                    samples.addAll(extractLengthBasedContext(sentences, baseContextSize));
                    break;
                case COMPLEXITY_BASED:
                    samples.addAll(extractComplexityBasedContext(sentences, baseContextSize));
                    break;
                case ENTROPY_BASED:
                    samples.addAll(extractEntropyBasedContext(sentences, baseContextSize));
                    break;
            }
        }
        return samples;
    }

    private List<ExtractedSample> extractFixedContext(List<String> sentences, int contextSize) {
        List<ExtractedSample> samples = new ArrayList<>();
        for (int i = contextSize; i < sentences.size(); i++) {
            StringBuilder ctx = new StringBuilder();
            for (int j = i - contextSize; j < i; j++) {
                if (ctx.length() > 0) ctx.append(" ");
                ctx.append(sentences.get(j));
            }
            samples.add(new ExtractedSample(ctx.toString(), sentences.get(i)));
        }
        return samples;
    }

    private List<ExtractedSample> extractLengthBasedContext(List<String> sentences, int baseContextSize) {
        List<ExtractedSample> samples = new ArrayList<>();
        int i = 0;
        
        while (i < sentences.size()) {
            int contextSize = baseContextSize;
            double avgLength = 0;
            int count = 0;
            
            for (int j = Math.max(0, i - baseContextSize * 2); j < i; j++) {
                avgLength += sentences.get(j).length();
                count++;
            }
            
            if (count > 0) {
                avgLength /= count;
                if (avgLength > 50) {
                    contextSize = Math.max(1, baseContextSize - 1);
                } else if (avgLength < 20) {
                    contextSize = baseContextSize + 1;
                }
            }
            
            if (i >= contextSize) {
                StringBuilder ctx = new StringBuilder();
                for (int j = i - contextSize; j < i; j++) {
                    if (ctx.length() > 0) ctx.append(" ");
                    ctx.append(sentences.get(j));
                }
                samples.add(new ExtractedSample(ctx.toString(), sentences.get(i)));
            }
            i++;
        }
        return samples;
    }

    private List<ExtractedSample> extractComplexityBasedContext(List<String> sentences, int baseContextSize) {
        List<ExtractedSample> samples = new ArrayList<>();
        
        for (int i = baseContextSize; i < sentences.size(); i++) {
            int contextSize = baseContextSize;
            double complexity = 0;
            
            for (int j = Math.max(0, i - baseContextSize); j < i; j++) {
                String sent = sentences.get(j);
                complexity += calculateComplexity(sent);
            }
            complexity /= Math.min(baseContextSize, i);
            
            if (complexity > 2.0) {
                contextSize = Math.max(1, baseContextSize - 1);
            } else if (complexity < 0.8) {
                contextSize = baseContextSize + 1;
            }
            
            StringBuilder ctx = new StringBuilder();
            for (int j = i - contextSize; j < i; j++) {
                if (ctx.length() > 0) ctx.append(" ");
                ctx.append(sentences.get(j));
            }
            samples.add(new ExtractedSample(ctx.toString(), sentences.get(i)));
        }
        return samples;
    }

    private double calculateComplexity(String sentence) {
        double complexity = 0;
        int wordCount = sentence.split("\\s+").length;
        int charCount = sentence.length();
        
        complexity += (double) charCount / Math.max(1, wordCount);
        
        String[] clauses = sentence.split("[，,。！？!？]");
        complexity += clauses.length * 0.3;
        
        int uniqueWords = new HashSet<>(Arrays.asList(sentence.toLowerCase().split("\\s+"))).size();
        complexity += (double) uniqueWords / Math.max(1, wordCount);
        
        return complexity / 3.0;
    }

    private List<ExtractedSample> extractEntropyBasedContext(List<String> sentences, int baseContextSize) {
        List<ExtractedSample> samples = new ArrayList<>();
        
        for (int i = baseContextSize; i < sentences.size(); i++) {
            int contextSize = baseContextSize;
            double entropyDelta = 0;
            
            if (i >= baseContextSize * 2) {
                double prevEntropy = calculateLocalEntropy(sentences, i - baseContextSize, baseContextSize);
                double currEntropy = calculateLocalEntropy(sentences, i, baseContextSize);
                entropyDelta = Math.abs(currEntropy - prevEntropy);
            }
            
            if (entropyDelta > 0.5) {
                contextSize = Math.max(1, baseContextSize - 1);
            } else if (entropyDelta < 0.2) {
                contextSize = baseContextSize + 1;
            }
            
            StringBuilder ctx = new StringBuilder();
            for (int j = i - contextSize; j < i; j++) {
                if (ctx.length() > 0) ctx.append(" ");
                ctx.append(sentences.get(j));
            }
            samples.add(new ExtractedSample(ctx.toString(), sentences.get(i)));
        }
        return samples;
    }

    private double calculateLocalEntropy(List<String> sentences, int startIdx, int windowSize) {
        Map<String, Integer> charFreq = new HashMap<>();
        int totalChars = 0;
        
        for (int i = startIdx; i < Math.min(startIdx + windowSize, sentences.size()); i++) {
            String sent = sentences.get(i);
            for (char c : sent.toCharArray()) {
                charFreq.merge(String.valueOf(c), 1, Integer::sum);
                totalChars++;
            }
        }
        
        if (totalChars == 0) return 0;
        
        double entropy = 0;
        for (int count : charFreq.values()) {
            double p = (double) count / totalChars;
            if (p > 0) {
                entropy -= p * (Math.log(p) / Math.log(2));
            }
        }
        
        return entropy;
    }

    // ========== 情感分析 ==========

    private List<ExtractedSample> extractSentiment(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();

        for (String text : rawTexts) {
            List<String> sentences = splitSentences(text);
            for (String sentence : sentences) {
                if (sentence.length() < 3) continue;
                String sentiment = analyzeSentiment(sentence);
                samples.add(new ExtractedSample(sentence, sentiment));
            }
        }
        return samples;
    }

    private String analyzeSentiment(String text) {
        String lower = text.toLowerCase();
        int posScore = 0, negScore = 0;

        for (String w : POSITIVE_WORDS) {
            if (lower.contains(w)) posScore++;
        }
        for (String w : NEGATIVE_WORDS) {
            if (lower.contains(w)) negScore++;
        }

        if (posScore > negScore) return "正面";
        if (negScore > posScore) return "负面";
        return "中性";
    }

    // ========== 摘要提取 ==========

    private List<ExtractedSample> extractSummary(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();

        for (String text : rawTexts) {
            List<String> paragraphs = splitParagraphs(text);
            if (paragraphs.size() < 3) continue;

            String fullText = String.join(" ", paragraphs);
            Set<String> keyPhrases = extractKeyPhrases(fullText, 5);

            for (int i = 0; i < paragraphs.size(); i++) {
                String p = paragraphs.get(i).trim();
                if (p.length() < 20) continue;

                String summary = extractiveSummary(p, Math.max(5, p.length() / 3));
                if (summary.length() >= 5 && summary.length() < p.length()) {
                    samples.add(new ExtractedSample(p, summary));
                }
            }
        }
        return samples;
    }

    private String extractiveSummary(String text, int maxLen) {
        List<String> sentences = splitSentences(text);
        if (sentences.size() <= 1) {
            return text.length() > maxLen ? text.substring(0, maxLen) : text;
        }

        Set<String> keywords = extractKeyPhrases(text, 5);
        double[] scores = new double[sentences.size()];
        for (int i = 0; i < sentences.size(); i++) {
            for (String kw : keywords) {
                if (sentences.get(i).toLowerCase().contains(kw)) {
                    scores[i] += 1.0;
                }
            }
            if (i == 0) scores[i] += 0.5;
        }

        StringBuilder summary = new StringBuilder();
        for (int i = 0; i < sentences.size(); i++) {
            if (scores[i] > 0) {
                if (summary.length() > 0) summary.append(" ");
                summary.append(sentences.get(i));
                if (summary.length() >= maxLen) break;
            }
        }

        if (summary.length() == 0 && !sentences.isEmpty()) {
            summary.append(sentences.get(0));
        }

        String result = summary.toString();
        return result.length() > maxLen ? result.substring(0, maxLen) : result;
    }

    // ========== 关键词预测 ==========

    private List<ExtractedSample> extractKeywordPredict(List<String> rawTexts) {
        List<ExtractedSample> samples = new ArrayList<>();
        int topK = getParamInt("topK", 3);

        for (String text : rawTexts) {
            List<String> paragraphs = splitParagraphs(text);
            for (String p : paragraphs) {
                if (p.length() < 20) continue;

                Set<String> keywords = extractKeyPhrases(p, topK);
                if (keywords.isEmpty()) continue;

                String keywordStr = String.join(", ", keywords);
                samples.add(new ExtractedSample(p, keywordStr));
            }
        }
        return samples;
    }

    private Set<String> extractKeyPhrases(String text, int topK) {
        Map<String, Integer> wordFreq = new LinkedHashMap<>();
        String cleaned = text.toLowerCase().replaceAll("[^\\p{L}\\p{N}\\s]", " ");
        String[] words = cleaned.split("\\s+");

        Set<String> stopWords = new HashSet<>(Arrays.asList(
                "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
                "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
                "for", "of", "and", "or", "but", "with", "this", "that", "it", "be"
        ));

        for (String w : words) {
            if (w.length() >= 2 && !stopWords.contains(w)) {
                wordFreq.merge(w, 1, Integer::sum);
            }
        }

        if (text.chars().filter(c -> c >= 0x4E00 && c <= 0x9FFF).count() > words.length / 2) {
            for (int ngram = 2; ngram <= 4; ngram++) {
                String t = text.toLowerCase().replaceAll("[^\\p{L}\\p{N}]", "");
                for (int i = 0; i <= t.length() - ngram; i++) {
                    String ng = t.substring(i, i + ngram);
                    boolean allChinese = ng.chars().allMatch(c -> c >= 0x4E00 && c <= 0x9FFF);
                    if (allChinese && !stopWords.contains(ng)) {
                        wordFreq.merge(ng, 1, Integer::sum);
                    }
                }
            }
        }

        return wordFreq.entrySet().stream()
                .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    // ========== 数据增强 ==========

    private List<ExtractedSample> extractAugment(List<String> rawTexts) {
        List<ExtractedSample> baseSamples = new ArrayList<>();
        int augmentFactor = getParamInt("factor", 3);

        for (String text : rawTexts) {
            List<String> sentences = splitSentences(text);
            for (String s : sentences) {
                if (s.length() < 5) continue;
                String sentiment = analyzeSentiment(s);
                baseSamples.add(new ExtractedSample(s, sentiment));

                for (int a = 0; a < augmentFactor; a++) {
                    String augmented = augmentText(s, a);
                    if (!augmented.equals(s) && augmented.length() >= 3) {
                        baseSamples.add(new ExtractedSample(augmented, sentiment));
                    }
                }
            }
        }
        return baseSamples;
    }

    private String augmentText(String text, int variant) {
        switch (variant % 4) {
            case 0: return synonymReplace(text);
            case 1: return randomDelete(text);
            case 2: return randomSwap(text);
            case 3: return insertNoise(text);
            default: return text;
        }
    }

    private String synonymReplace(String text) {
        String result = text;
        List<Map.Entry<String, String[]>> entries = new ArrayList<>(SYNONYM_MAP.entrySet());
        Collections.shuffle(entries, rng);

        int replaced = 0;
        for (Map.Entry<String, String[]> entry : entries) {
            if (result.contains(entry.getKey())) {
                String[] synonyms = entry.getValue();
                String synonym = synonyms[rng.nextInt(synonyms.length)];
                result = result.replaceFirst(Pattern.quote(entry.getKey()), synonym);
                replaced++;
                if (replaced >= 2) break;
            }
        }
        return result;
    }

    private String randomDelete(String text) {
        char[] chars = text.toCharArray();
        if (chars.length <= 3) return text;

        int deleteCount = Math.max(1, chars.length / 10);
        Set<Integer> deleted = new HashSet<>();
        for (int d = 0; d < deleteCount; d++) {
            int idx = rng.nextInt(chars.length);
            if (!Character.isHighSurrogate(chars[idx]) && !Character.isLowSurrogate(chars[idx])) {
                deleted.add(idx);
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < chars.length; i++) {
            if (!deleted.contains(i)) sb.append(chars[i]);
        }
        return sb.toString();
    }

    private String randomSwap(String text) {
        List<String> tokens = Arrays.asList(text.split("(?<=\\s)|(?=\\s)"));
        if (tokens.size() < 4) return text;

        List<String> mutable = new ArrayList<>(tokens);
        int swapCount = Math.max(1, mutable.size() / 8);
        for (int s = 0; s < swapCount; s++) {
            int i = rng.nextInt(mutable.size());
            int j = rng.nextInt(mutable.size());
            Collections.swap(mutable, i, j);
        }
        return String.join("", mutable);
    }

    private String insertNoise(String text) {
        String[] noise = {"", " ", " "};
        StringBuilder sb = new StringBuilder(text);
        int insertCount = Math.max(1, sb.length() / 15);
        for (int i = 0; i < insertCount; i++) {
            int pos = rng.nextInt(sb.length());
            sb.insert(pos, noise[rng.nextInt(noise.length)]);
        }
        return sb.toString().trim();
    }

    // ========== 工具方法 ==========

    private List<String> splitParagraphs(String text) {
        String[] parts = text.split("\\n\\s*\\n|\\n");
        List<String> result = new ArrayList<>();
        for (String p : parts) {
            String trimmed = p.trim();
            if (trimmed.length() >= 5) result.add(trimmed);
        }
        return result;
    }

    private List<String> splitSentences(String text) {
        String normalized = text.replaceAll("\\r\\n", "\n").replaceAll("\\r", "\n");
        String[] parts = normalized.split("(?<=[。！？\\.\\!\\?])\\s*|\\n+");
        List<String> result = new ArrayList<>();
        for (String s : parts) {
            String trimmed = s.trim();
            if (trimmed.length() >= 3) result.add(trimmed);
        }
        return result;
    }

    private boolean containsChinese(String text) {
        return text.chars().anyMatch(c -> c >= 0x4E00 && c <= 0x9FFF);
    }

    private boolean containsEnglish(String text) {
        return text.chars().anyMatch(c -> (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    }

    public Strategy getStrategy() { return strategy; }

    public static String[] strategyLabels() {
        Strategy[] values = Strategy.values();
        String[] labels = new String[values.length];
        for (int i = 0; i < values.length; i++) labels[i] = values[i].label;
        return labels;
    }

    public static Strategy fromLabel(String label) {
        for (Strategy s : Strategy.values()) {
            if (s.label.equals(label)) return s;
        }
        return Strategy.ARTICLE_TOPIC;
    }
}

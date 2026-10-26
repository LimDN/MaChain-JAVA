package machain.data;

import java.io.*;
import java.util.*;
import java.util.regex.*;

public class TextProcessor implements Serializable {
    private static final long serialVersionUID = 2L;

    public enum VectorizeMethod {
        BAG_OF_WORDS,
        TF_IDF,
        CHAR_NGRAM,
        EMBEDDING_SEQUENCE
    }

    private Map<String, Integer> vocabulary;
    private Map<Integer, String> reverseVocabulary;
    private Map<String, Double> idf;
    private VectorizeMethod method;
    private int maxVocabSize;
    private int ngramSize;
    private boolean built = false;
    private boolean preservePunctuation = false;

    public TextProcessor(VectorizeMethod method, int maxVocabSize) {
        this.method = method;
        this.maxVocabSize = maxVocabSize;
        this.ngramSize = method == VectorizeMethod.CHAR_NGRAM ? 3 : 1;
        this.vocabulary = new LinkedHashMap<>();
        this.reverseVocabulary = new HashMap<>();
        this.idf = new HashMap<>();
    }

    public void buildVocabulary(List<String> texts) {
        vocabulary.clear();
        reverseVocabulary.clear();
        idf.clear();

        Map<String, Integer> wordFreq = new HashMap<>();
        int docCount = texts.size();

        for (String text : texts) {
            Set<String> uniqueTokens = new HashSet<>();
            List<String> tokens = tokenize(text);

            for (String token : tokens) {
                wordFreq.merge(token, 1, Integer::sum);
                uniqueTokens.add(token);
            }

            for (String token : uniqueTokens) {
                idf.merge(token, 1.0, Double::sum);
            }
        }

        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(wordFreq.entrySet());
        sorted.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        int limit = Math.min(maxVocabSize, sorted.size());
        for (int i = 0; i < limit; i++) {
            String word = sorted.get(i).getKey();
            vocabulary.put(word, i);
            reverseVocabulary.put(i, word);
        }

        for (Map.Entry<String, Double> entry : idf.entrySet()) {
            entry.setValue(Math.log((docCount + 1.0) / (entry.getValue() + 1.0)) + 1.0);
        }

        built = true;
    }

    public List<String> tokenize(String text) {
        text = text.toLowerCase().trim();
        
        if (preservePunctuation) {
            text = text.replaceAll("[^\\p{L}\\p{N}\\p{P}\\s]", " ");
        } else {
            text = text.replaceAll("[^\\p{L}\\p{N}\\s]", " ");
        }

        if (method == VectorizeMethod.CHAR_NGRAM) {
            return extractCharNgrams(text);
        } else {
            return extractWordTokens(text);
        }
    }

    public boolean isPreservePunctuation() {
        return preservePunctuation;
    }

    public void setPreservePunctuation(boolean preserve) {
        this.preservePunctuation = preserve;
    }

    private List<String> extractWordTokens(String text) {
        List<String> tokens = new ArrayList<>();
        
        if (preservePunctuation) {
            StringBuilder current = new StringBuilder();
            for (char c : text.toCharArray()) {
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
            String[] words = text.split("\\s+");
            for (String word : words) {
                if (!word.isEmpty()) {
                    tokens.add(word);
                }
            }
        }
        
        return tokens;
    }

    private List<String> extractCharNgrams(String text) {
        List<String> ngrams = new ArrayList<>();
        for (int i = 0; i <= text.length() - ngramSize; i++) {
            ngrams.add(text.substring(i, i + ngramSize));
        }
        return ngrams;
    }

    public double[] vectorize(String text) {
        if (!built) {
            throw new IllegalStateException("必须先调用 buildVocabulary()");
        }

        double[] vector = new double[vocabulary.size()];
        List<String> tokens = tokenize(text);
        Map<String, Integer> tokenCounts = new HashMap<>();

        for (String token : tokens) {
            tokenCounts.merge(token, 1, Integer::sum);
        }

        switch (method) {
            case BAG_OF_WORDS:
                for (Map.Entry<String, Integer> entry : tokenCounts.entrySet()) {
                    Integer idx = vocabulary.get(entry.getKey());
                    if (idx != null) {
                        vector[idx] = entry.getValue();
                    }
                }
                break;

            case TF_IDF:
                int totalTokens = tokens.size();
                for (Map.Entry<String, Integer> entry : tokenCounts.entrySet()) {
                    Integer idx = vocabulary.get(entry.getKey());
                    if (idx != null) {
                        double tf = entry.getValue() / (double) totalTokens;
                        double idfValue = idf.getOrDefault(entry.getKey(), 1.0);
                        vector[idx] = tf * idfValue;
                    }
                }
                break;

            case CHAR_NGRAM:
                for (Map.Entry<String, Integer> entry : tokenCounts.entrySet()) {
                    Integer idx = vocabulary.get(entry.getKey());
                    if (idx != null) {
                        vector[idx] = entry.getValue();
                    }
                }
                break;
        }

        return vector;
    }

    public int getVectorSize() {
        return vocabulary.size();
    }

    public Map<String, Integer> getVocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }

    public boolean isBuilt() {
        return built;
    }

    public VectorizeMethod getMethod() {
        return method;
    }

    public int[] textToSequence(String text, int maxLength) {
        if (!built) {
            throw new IllegalStateException("必须先调用 buildVocabulary()");
        }

        List<String> tokens = tokenize(text);
        int[] sequence = new int[maxLength];
        Arrays.fill(sequence, 0);

        int len = Math.min(tokens.size(), maxLength);
        for (int i = 0; i < len; i++) {
            Integer idx = vocabulary.get(tokens.get(i));
            sequence[i] = (idx != null) ? idx + 1 : 0;
        }

        return sequence;
    }

    public int getVocabSize() {
        return vocabulary.size() + 1;
    }

    public String sequenceToText(int[] sequence) {
        StringBuilder sb = new StringBuilder();
        for (int idx : sequence) {
            if (idx == 0) continue;
            String word = reverseVocabulary.get(idx - 1);
            if (word != null) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(word);
            }
        }
        return sb.toString();
    }

    public void saveVocabulary(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(vocabulary);
            oos.writeObject(reverseVocabulary);
            oos.writeObject(idf);
            oos.writeObject(method);
            oos.writeInt(maxVocabSize);
            oos.writeInt(ngramSize);
            oos.writeBoolean(built);
            oos.writeBoolean(preservePunctuation);
        }
    }

    public void loadVocabulary(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            vocabulary = (Map<String, Integer>) ois.readObject();
            reverseVocabulary = (Map<Integer, String>) ois.readObject();
            idf = (Map<String, Double>) ois.readObject();
            method = (VectorizeMethod) ois.readObject();
            maxVocabSize = ois.readInt();
            ngramSize = ois.readInt();
            built = ois.readBoolean();
            preservePunctuation = ois.readBoolean();
        }
    }
}

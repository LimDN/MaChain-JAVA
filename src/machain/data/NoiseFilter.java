package machain.data;

import java.util.*;
import java.util.stream.*;

public class NoiseFilter {
    
    private int filterLevel;
    
    public NoiseFilter(int level) {
        this.filterLevel = Math.max(0, Math.min(10, level));
    }
    
    public List<SampleExtractor.ExtractedSample> filter(List<SampleExtractor.ExtractedSample> samples) {
        if (filterLevel == 0 || samples.isEmpty()) {
            return samples;
        }
        
        List<SampleExtractor.ExtractedSample> filtered = new ArrayList<>(samples);
        
        filtered = removeEmptyAndTooShort(filtered);
        filtered = removeDuplicates(filtered);
        filtered = removeInvalidCharacters(filtered);
        
        if (filterLevel >= 3) {
            filtered = removeLengthOutliers(filtered);
        }
        
        if (filterLevel >= 5) {
            filtered = removeLowDiversity(filtered);
        }
        
        if (filterLevel >= 7) {
            filtered = removeStatisticalOutliers(filtered);
        }
        
        if (filterLevel >= 9) {
            filtered = removeRepetitivePatterns(filtered);
        }
        
        return filtered;
    }
    
    public double[][] filterInputs(double[][] inputs, int filterLevel) {
        if (filterLevel == 0 || inputs.length == 0) {
            return inputs;
        }
        
        List<Integer> validIndices = new ArrayList<>();
        
        for (int i = 0; i < inputs.length; i++) {
            if (isValidRow(inputs[i], filterLevel)) {
                validIndices.add(i);
            }
        }
        
        double[][] filtered = new double[validIndices.size()][inputs[0].length];
        for (int i = 0; i < validIndices.size(); i++) {
            filtered[i] = inputs[validIndices.get(i)];
        }
        
        return filtered;
    }
    
    private boolean isValidRow(double[] row, int level) {
        double sum = 0;
        int nonZeroCount = 0;
        
        for (double v : row) {
            sum += Math.abs(v);
            if (Math.abs(v) > 1e-6) nonZeroCount++;
        }
        
        if (sum < 1e-6) return false;
        
        if (level >= 3) {
            double ratio = (double) nonZeroCount / row.length;
            if (ratio < 0.01 || ratio > 0.99) return false;
        }
        
        if (level >= 5) {
            double mean = sum / row.length;
            double variance = 0;
            for (double v : row) {
                variance += (v - mean) * (v - mean);
            }
            variance /= row.length;
            double stdDev = Math.sqrt(variance);
            
            if (stdDev < 1e-6) return false;
        }
        
        if (level >= 7) {
            double mean = sum / row.length;
            double threshold = mean * 3;
            for (double v : row) {
                if (Math.abs(v) > threshold) return false;
            }
        }
        
        return true;
    }
    
    private List<SampleExtractor.ExtractedSample> removeEmptyAndTooShort(List<SampleExtractor.ExtractedSample> samples) {
        int minLength = filterLevel >= 5 ? 10 : (filterLevel >= 3 ? 5 : 3);
        return samples.stream()
            .filter(s -> s.inputText != null && s.inputText.trim().length() >= minLength)
            .collect(Collectors.toList());
    }
    
    private List<SampleExtractor.ExtractedSample> removeDuplicates(List<SampleExtractor.ExtractedSample> samples) {
        if (filterLevel < 2) return samples;
        
        Set<String> seen = new HashSet<>();
        List<SampleExtractor.ExtractedSample> unique = new ArrayList<>();
        
        for (SampleExtractor.ExtractedSample s : samples) {
            String key = s.inputText.trim().toLowerCase();
            if (!seen.contains(key)) {
                seen.add(key);
                unique.add(s);
            }
        }
        
        return unique;
    }
    
    private List<SampleExtractor.ExtractedSample> removeInvalidCharacters(List<SampleExtractor.ExtractedSample> samples) {
        if (filterLevel < 2) return samples;
        
        double maxInvalidRatio = filterLevel >= 7 ? 0.05 : (filterLevel >= 5 ? 0.1 : 0.2);
        
        return samples.stream()
            .filter(s -> {
                String text = s.inputText;
                int invalidCount = 0;
                for (char c : text.toCharArray()) {
                    if (!Character.isLetterOrDigit(c) && !Character.isWhitespace(c) 
                        && !isPunctuation(c)) {
                        invalidCount++;
                    }
                }
                double ratio = (double) invalidCount / text.length();
                return ratio <= maxInvalidRatio;
            })
            .collect(Collectors.toList());
    }
    
    private boolean isPunctuation(char c) {
        return Character.getType(c) == Character.CONNECTOR_PUNCTUATION
            || Character.getType(c) == Character.DASH_PUNCTUATION
            || Character.getType(c) == Character.START_PUNCTUATION
            || Character.getType(c) == Character.END_PUNCTUATION
            || Character.getType(c) == Character.INITIAL_QUOTE_PUNCTUATION
            || Character.getType(c) == Character.FINAL_QUOTE_PUNCTUATION
            || Character.getType(c) == Character.OTHER_PUNCTUATION;
    }
    
    private List<SampleExtractor.ExtractedSample> removeLengthOutliers(List<SampleExtractor.ExtractedSample> samples) {
        if (samples.isEmpty()) return samples;
        
        List<Integer> lengths = samples.stream()
            .map(s -> s.inputText.length())
            .collect(Collectors.toList());
        
        double mean = lengths.stream().mapToInt(Integer::intValue).average().orElse(0);
        double stdDev = Math.sqrt(lengths.stream()
            .mapToDouble(l -> (l - mean) * (l - mean))
            .average().orElse(0));
        
        double threshold = filterLevel >= 7 ? 1.5 : 2.0;
        double minLen = mean - threshold * stdDev;
        double maxLen = mean + threshold * stdDev;
        
        return samples.stream()
            .filter(s -> {
                int len = s.inputText.length();
                return len >= minLen && len <= maxLen;
            })
            .collect(Collectors.toList());
    }
    
    private List<SampleExtractor.ExtractedSample> removeLowDiversity(List<SampleExtractor.ExtractedSample> samples) {
        double minDiversity = filterLevel >= 9 ? 0.3 : 0.2;
        
        return samples.stream()
            .filter(s -> {
                String text = s.inputText.toLowerCase();
                Set<Character> uniqueChars = new HashSet<>();
                for (char c : text.toCharArray()) {
                    uniqueChars.add(c);
                }
                double diversity = (double) uniqueChars.size() / text.length();
                return diversity >= minDiversity;
            })
            .collect(Collectors.toList());
    }
    
    private List<SampleExtractor.ExtractedSample> removeStatisticalOutliers(List<SampleExtractor.ExtractedSample> samples) {
        if (samples.size() < 10) return samples;
        
        List<Double> complexities = samples.stream()
            .map(this::calculateComplexity)
            .collect(Collectors.toList());
        
        double mean = complexities.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stdDev = Math.sqrt(complexities.stream()
            .mapToDouble(c -> (c - mean) * (c - mean))
            .average().orElse(0));
        
        double threshold = filterLevel >= 9 ? 1.0 : 1.5;
        double minComplexity = mean - threshold * stdDev;
        double maxComplexity = mean + threshold * stdDev;
        
        List<SampleExtractor.ExtractedSample> filtered = new ArrayList<>();
        for (int i = 0; i < samples.size(); i++) {
            double complexity = complexities.get(i);
            if (complexity >= minComplexity && complexity <= maxComplexity) {
                filtered.add(samples.get(i));
            }
        }
        
        return filtered;
    }
    
    private double calculateComplexity(SampleExtractor.ExtractedSample sample) {
        String text = sample.inputText;
        int wordCount = text.split("\\s+").length;
        int uniqueWords = new HashSet<>(Arrays.asList(text.toLowerCase().split("\\s+"))).size();
        return (double) uniqueWords / Math.max(1, wordCount);
    }
    
    private List<SampleExtractor.ExtractedSample> removeRepetitivePatterns(List<SampleExtractor.ExtractedSample> samples) {
        return samples.stream()
            .filter(s -> {
                String text = s.inputText;
                if (text.length() < 20) return true;
                
                String repeated = findRepetitivePattern(text);
                if (repeated != null && repeated.length() > 3) {
                    int count = countOccurrences(text, repeated);
                    double ratio = (double) (count * repeated.length()) / text.length();
                    return ratio < 0.5;
                }
                return true;
            })
            .collect(Collectors.toList());
    }
    
    private String findRepetitivePattern(String text) {
        for (int len = 3; len <= text.length() / 3; len++) {
            String pattern = text.substring(0, len);
            if (isRepetitive(text, pattern)) {
                return pattern;
            }
        }
        return null;
    }
    
    private boolean isRepetitive(String text, String pattern) {
        int count = 0;
        int idx = 0;
        while ((idx = text.indexOf(pattern, idx)) != -1) {
            count++;
            idx += pattern.length();
        }
        return count >= 3;
    }
    
    private int countOccurrences(String text, String pattern) {
        int count = 0;
        int idx = 0;
        while ((idx = text.indexOf(pattern, idx)) != -1) {
            count++;
            idx += pattern.length();
        }
        return count;
    }
    
    public int getFilterLevel() {
        return filterLevel;
    }
}

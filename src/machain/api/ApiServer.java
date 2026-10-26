package machain.api;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import machain.network.Network;
import machain.network.OutputMode;
import machain.data.TextProcessor;
import machain.generation.TextGenerator;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;

public class ApiServer {

    private HttpServer server;
    private int port;
    private boolean running;

    private volatile Network network;
    private volatile TextProcessor textProcessor;
    private volatile TextGenerator textGenerator;
    private volatile OutputMode outputMode = OutputMode.REGRESSION;
    private volatile machain.network.EnhancedNetwork enhancedNetwork;

    public ApiServer(int port) {
        this.port = port;
    }

    public void setModel(Network network, TextProcessor tp, OutputMode mode) {
        this.network = network;
        this.textProcessor = tp;
        this.outputMode = mode != null ? mode : OutputMode.REGRESSION;
    }

    public void setTextGenerator(TextGenerator gen) {
        this.textGenerator = gen;
    }

    public void setEnhancedNetwork(machain.network.EnhancedNetwork en) {
        this.enhancedNetwork = en;
    }

    public void start() throws IOException {
        if (running) return;
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/api/status", new StatusHandler());
        server.createContext("/api/predict", new PredictHandler());
        server.createContext("/api/generate", new GenerateHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        running = true;
    }

    public void stop() {
        if (!running) return;
        server.stop(0);
        running = false;
    }

    public boolean isRunning() { return running; }
    public int getPort() { return port; }

    private String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private void sendJson(HttpExchange ex, int code, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange ex, int code, String msg) throws IOException {
        sendJson(ex, code, "{\"error\":\"" + escapeJson(msg) + "\"}");
    }

    static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private double[] parseInputArray(String body) {
        String trimmed = body.trim();
        if (trimmed.startsWith("[")) {
            trimmed = trimmed.substring(1);
        }
        if (trimmed.endsWith("]")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        String[] parts = trimmed.split("[,\\s]+");
        double[] result = new double[parts.length];
        int count = 0;
        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            try {
                result[count++] = Double.parseDouble(p);
            } catch (NumberFormatException ignored) {}
        }
        double[] out = new double[count];
        System.arraycopy(result, 0, out, 0, count);
        return out;
    }

    private String extractJsonString(String json, String key) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return null;
        idx = json.indexOf(":", idx + pattern.length());
        if (idx < 0) return null;
        idx++;
        while (idx < json.length() && json.charAt(idx) == ' ') idx++;
        if (idx >= json.length()) return null;
        if (json.charAt(idx) == '"') {
            int end = json.indexOf('"', idx + 1);
            if (end < 0) return null;
            return json.substring(idx + 1, end);
        }
        int end = idx;
        while (end < json.length() && json.charAt(end) != ',' && json.charAt(end) != '}') end++;
        return json.substring(idx, end).trim();
    }

    private double extractJsonDouble(String json, String key, double def) {
        String val = extractJsonString(json, key);
        if (val == null) return def;
        try { return Double.parseDouble(val); } catch (Exception e) { return def; }
    }

    private boolean extractJsonBool(String json, String key, boolean def) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return def;
        idx = json.indexOf(":", idx + pattern.length());
        if (idx < 0) return def;
        String rest = json.substring(idx + 1).trim();
        if (rest.startsWith("true")) return true;
        if (rest.startsWith("false")) return false;
        return def;
    }

    private boolean isTextInput(String text) {
        return text != null && !text.isEmpty() && !text.matches("[\\d.,;\\s\\[\\]\\-]+");
    }

    private class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
                sendError(ex, 405, "Method not allowed");
                return;
            }
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"running\":true");
            sb.append(",\"port\":").append(port);

            if (network != null) {
                sb.append(",\"network\":{");
                sb.append("\"architecture\":\"").append(escapeJson(network.getArchitecture())).append("\"");
                sb.append(",\"parameters\":").append(network.countParameters());
                sb.append(",\"epochs\":").append(network.getTotalEpochs());
                sb.append(",\"inputDim\":").append(network.getLayerSizes()[0]);
                sb.append(",\"outputDim\":").append(network.getLayerSizes()[network.getLayerSizes().length - 1]);
                sb.append(",\"mode\":\"").append(outputMode.getDisplayName()).append("\"");
                sb.append("}");
            } else {
                sb.append(",\"network\":null");
            }

            sb.append(",\"textGenerator\":");
            if (textGenerator != null) {
                sb.append("\"").append(escapeJson(textGenerator.getStats())).append("\"");
            } else {
                sb.append("null");
            }

            sb.append(",\"enhancedNetwork\":");
            sb.append(enhancedNetwork != null ? "true" : "false");

            sb.append(",\"textProcessor\":");
            sb.append(textProcessor != null ? "true" : "false");

            sb.append("}");
            sendJson(ex, 200, sb.toString());
        }
    }

    private class PredictHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                sendError(ex, 405, "Method not allowed");
                return;
            }

            if (network == null) {
                sendError(ex, 503, "No model loaded");
                return;
            }

            String body = readBody(ex);
            if (body.isEmpty()) {
                sendError(ex, 400, "Empty body");
                return;
            }

            try {
                int expectedDim = network.getLayerSizes()[0];
                double[] input;

                String rawInput = extractJsonString(body, "input");
                if (rawInput != null && isTextInput(rawInput)) {
                    if (textProcessor == null) {
                        sendError(ex, 400, "TextProcessor not available for text input");
                        return;
                    }
                    input = textProcessor.vectorize(rawInput);
                    if (input.length != expectedDim) {
                        double[] padded = new double[expectedDim];
                        System.arraycopy(input, 0, padded, 0, Math.min(input.length, expectedDim));
                        input = padded;
                    }
                } else {
                    int inputStart = body.indexOf("\"input\"");
                    if (inputStart < 0) {
                        sendError(ex, 400, "Missing 'input' field");
                        return;
                    }
                    int colonIdx = body.indexOf(":", inputStart);
                    int arrStart = body.indexOf("[", colonIdx);
                    int arrEnd = body.indexOf("]", arrStart);
                    if (arrStart < 0 || arrEnd < 0) {
                        sendError(ex, 400, "Invalid input array");
                        return;
                    }
                    input = parseInputArray(body.substring(arrStart, arrEnd + 1));
                    if (input.length != expectedDim) {
                        double[] padded = new double[expectedDim];
                        System.arraycopy(input, 0, padded, 0, Math.min(input.length, expectedDim));
                        input = padded;
                    }
                }

                double[] output = network.forward(input);

                StringBuilder sb = new StringBuilder();
                sb.append("{");
                sb.append("\"output\":[");
                for (int i = 0; i < output.length; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.6f", output[i]));
                }
                sb.append("]");
                sb.append(",\"mode\":\"").append(outputMode.getDisplayName()).append("\"");

                if (outputMode == OutputMode.MULTICLASS_CLASSIFICATION) {
                    int bestIdx = 0;
                    double bestVal = output[0];
                    for (int i = 1; i < output.length; i++) {
                        if (output[i] > bestVal) {
                            bestVal = output[i];
                            bestIdx = i;
                        }
                    }
                    sb.append(",\"predictedClass\":").append(bestIdx);
                    sb.append(",\"confidence\":").append(String.format("%.4f", bestVal));
                }

                sb.append("}");
                sendJson(ex, 200, sb.toString());

            } catch (Exception e) {
                sendError(ex, 500, "Predict failed: " + e.getMessage());
            }
        }
    }

    private class GenerateHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                sendError(ex, 405, "Method not allowed");
                return;
            }

            if (textGenerator == null) {
                sendError(ex, 503, "TextGenerator not built. Call /api/status to check.");
                return;
            }

            String body = readBody(ex);
            try {
                String seed = extractJsonString(body, "seed");
                double temp = extractJsonDouble(body, "temperature", 1.0);
                int maxLen = (int) extractJsonDouble(body, "max_length", 100);
                boolean preservePunct = extractJsonBool(body, "preserve_punctuation", false);

                textGenerator.setTemperature(temp);
                textGenerator.setMaxGenLength(maxLen);
                textGenerator.setPreservePunctuation(preservePunct);

                String generated = textGenerator.generate(seed);

                sendJson(ex, 200, "{"
                        + "\"generated\":\"" + escapeJson(generated) + "\""
                        + ",\"seed\":\"" + escapeJson(seed != null ? seed : "") + "\""
                        + ",\"temperature\":" + String.format("%.2f", temp)
                        + ",\"max_length\":" + maxLen
                        + ",\"length\":" + generated.length()
                        + ",\"stats\":\"" + escapeJson(textGenerator.getStats()) + "\""
                        + "}");

            } catch (Exception e) {
                sendError(ex, 500, "Generate failed: " + e.getMessage());
            }
        }
    }
}

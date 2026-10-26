package machain.io;

import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class SampleScraper {

    public interface ScraperListener {
        void onSampleCaptured(String source, String text);
        void onError(String source, String message);
    }

    private ScraperListener listener;
    private volatile boolean monitoring = false;
    private Thread monitorThread;
    private String lastClipboard = "";
    private WatchService folderWatcher;

    public void setListener(ScraperListener listener) {
        this.listener = listener;
    }

    public String fetchUrl(String urlStr) {
        try {
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestProperty("User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(15000);
            conn.setRequestMethod("GET");

            int code = conn.getResponseCode();
            if (code != 200) return "[HTTP " + code + "]";

            String encoding = conn.getContentEncoding();
            String contentType = conn.getContentType();
            String charset = "UTF-8";
            if (contentType != null) {
                Matcher m = Pattern.compile("charset=([\\w-]+)").matcher(contentType);
                if (m.find()) charset = m.group(1);
            }

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), charset))) {
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append("\n");
                }
                return stripHtml(sb.toString());
            }
        } catch (Exception e) {
            if (listener != null) listener.onError(urlStr, e.getMessage());
            return "[错误] " + e.getMessage();
        }
    }

    public List<String> fetchMultipleUrls(List<String> urls) {
        List<String> results = new ArrayList<>();
        for (String u : urls) {
            String text = fetchUrl(u);
            if (!text.startsWith("[")) {
                results.add(text);
                if (listener != null) listener.onSampleCaptured(u, text);
            }
        }
        return results;
    }

    public String getClipboardText() {
        try {
            Clipboard cb = Toolkit.getDefaultToolkit().getSystemClipboard();
            if (cb.isDataFlavorAvailable(DataFlavor.stringFlavor)) {
                return (String) cb.getData(DataFlavor.stringFlavor);
            }
        } catch (Exception e) {
            // clipboard unavailable
        }
        return "";
    }

    public void startClipboardMonitor(long intervalMs) {
        if (monitoring) return;
        monitoring = true;
        lastClipboard = getClipboardText();
        monitorThread = new Thread(() -> {
            while (monitoring) {
                try {
                    Thread.sleep(intervalMs);
                    String current = getClipboardText();
                    if (!current.isEmpty() && !current.equals(lastClipboard)) {
                        lastClipboard = current;
                        if (listener != null) {
                            listener.onSampleCaptured("剪贴板", current);
                        }
                    }
                } catch (InterruptedException e) {
                    break;
                }
            }
        }, "ClipboardMonitor");
        monitorThread.setDaemon(true);
        monitorThread.start();
    }

    public void stopClipboardMonitor() {
        monitoring = false;
        if (monitorThread != null) monitorThread.interrupt();
    }

    public boolean isMonitoring() {
        return monitoring;
    }

    public void startFolderWatch(String folderPath) {
        try {
            Path dir = Paths.get(folderPath);
            if (!Files.isDirectory(dir)) return;
            folderWatcher = FileSystems.getDefault().newWatchService();
            dir.register(folderWatcher,
                    StandardWatchEventKinds.ENTRY_CREATE,
                    StandardWatchEventKinds.ENTRY_MODIFY);

            Thread watchThread = new Thread(() -> {
                while (folderWatcher != null) {
                    try {
                        WatchKey key = folderWatcher.take();
                        for (WatchEvent<?> event : key.pollEvents()) {
                            if (event.kind() == StandardWatchEventKinds.OVERFLOW) continue;
                            Path fileName = dir.resolve((Path) event.context()).toAbsolutePath();
                            String name = fileName.toString().toLowerCase();
                            if (name.endsWith(".txt") || name.endsWith(".csv")
                                    || name.endsWith(".md") || name.endsWith(".tsv")) {
                                try {
                                    String content = Files.readString(fileName, StandardCharsets.UTF_8);
                                    if (listener != null) {
                                        listener.onSampleCaptured(fileName.toString(), content);
                                    }
                                } catch (Exception ex) {
                                    if (listener != null) {
                                        listener.onError(fileName.toString(), ex.getMessage());
                                    }
                                }
                            }
                        }
                        key.reset();
                    } catch (InterruptedException e) {
                        break;
                    } catch (ClosedWatchServiceException e) {
                        break;
                    }
                }
            }, "FolderWatcher");
            watchThread.setDaemon(true);
            watchThread.start();
        } catch (Exception e) {
            if (listener != null) listener.onError(folderPath, e.getMessage());
        }
    }

    public void stopFolderWatch() {
        try {
            if (folderWatcher != null) {
                folderWatcher.close();
                folderWatcher = null;
            }
        } catch (Exception ignored) {}
    }

    public static String stripHtml(String html) {
        String text = html.replaceAll("<script[^>]*>[\\s\\S]*?</script>", " ");
        text = text.replaceAll("<style[^>]*>[\\s\\S]*?</style>", " ");
        text = text.replaceAll("<[^>]+>", " ");
        text = text.replaceAll("&nbsp;", " ");
        text = text.replaceAll("&lt;", "<");
        text = text.replaceAll("&gt;", ">");
        text = text.replaceAll("&amp;", "&");
        text = text.replaceAll("&quot;", "\"");
        text = text.replaceAll("&#\\d+;", " ");
        text = text.replaceAll("[ \\t]+", " ");
        text = text.replaceAll("\\n{3,}", "\n\n");
        return text.trim();
    }

    public static List<String> splitIntoParagraphs(String text) {
        List<String> paragraphs = new ArrayList<>();
        for (String p : text.split("\\n\\s*\\n")) {
            String trimmed = p.trim();
            if (trimmed.length() > 10) {
                paragraphs.add(trimmed);
            }
        }
        return paragraphs;
    }

    public static List<String> splitIntoSentences(String text) {
        List<String> sentences = new ArrayList<>();
        for (String s : text.split("(?<=[。！？.!?])\\s*")) {
            String trimmed = s.trim();
            if (trimmed.length() > 2) {
                sentences.add(trimmed);
            }
        }
        return sentences;
    }
}

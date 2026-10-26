package machain.io;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class OutputExporter {

    public static void exportCSV(String path, String[] headers, List<double[]> rows) throws IOException {
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(path), StandardCharsets.UTF_8))) {
            if (headers != null) {
                pw.println(String.join(",", headers));
            }
            for (double[] row : rows) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.8f", row[i]));
                }
                pw.println(sb.toString());
            }
        }
    }

    public static void exportJSON(String path, String[] headers, List<double[]> rows) throws IOException {
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(path), StandardCharsets.UTF_8))) {
            pw.println("[");
            for (int r = 0; r < rows.size(); r++) {
                double[] row = rows.get(r);
                pw.print("  {");
                for (int i = 0; i < row.length; i++) {
                    if (i > 0) pw.print(", ");
                    String key = (headers != null && i < headers.length) ? headers[i] : "v" + i;
                    pw.print("\"" + key + "\":" + String.format("%.8f", row[i]));
                }
                pw.print("}");
                if (r < rows.size() - 1) pw.print(",");
                pw.println();
            }
            pw.println("]");
        }
    }

    public static void exportText(String path, List<String> lines) throws IOException {
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(path), StandardCharsets.UTF_8))) {
            for (String line : lines) {
                pw.println(line);
            }
        }
    }

    public static void exportTSV(String path, String[] headers, List<double[]> rows) throws IOException {
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(path), StandardCharsets.UTF_8))) {
            if (headers != null) {
                pw.println(String.join("\t", headers));
            }
            for (double[] row : rows) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    if (i > 0) sb.append("\t");
                    sb.append(String.format("%.8f", row[i]));
                }
                pw.println(sb.toString());
            }
        }
    }

    public static void exportBatch(String dir, String baseName, String[] headers,
                                    List<double[]> rows, String... formats) throws IOException {
        File folder = new File(dir);
        if (!folder.exists()) folder.mkdirs();
        for (String fmt : formats) {
            String fileName = baseName + "." + fmt.toLowerCase();
            String fullPath = new File(folder, fileName).getAbsolutePath();
            switch (fmt.toUpperCase()) {
                case "CSV": exportCSV(fullPath, headers, rows); break;
                case "JSON": exportJSON(fullPath, headers, rows); break;
                case "TSV": exportTSV(fullPath, headers, rows); break;
            }
        }
    }

    public static String formatAsTable(String[] headers, List<double[]> rows) {
        StringBuilder sb = new StringBuilder();
        if (headers != null) {
            sb.append(String.join("\t", headers)).append("\n");
            sb.append("-".repeat(headers.length * 12)).append("\n");
        }
        for (double[] row : rows) {
            for (int i = 0; i < row.length; i++) {
                if (i > 0) sb.append("\t");
                sb.append(String.format("%.6f", row[i]));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}

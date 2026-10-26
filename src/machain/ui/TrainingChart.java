package machain.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.*;
import java.util.List;

public class TrainingChart extends JPanel {

    public enum MetricType {
        LOSS("Loss", Color.RED),
        VAL_LOSS("Val Loss", Color.BLUE),
        ACCURACY("Accuracy", Color.GREEN),
        LEARNING_RATE("LR", Color.ORANGE);

        final String label;
        final Color color;

        MetricType(String label, Color color) {
            this.label = label;
            this.color = color;
        }
    }

    private final Map<MetricType, List<Double>> data = new LinkedHashMap<>();
    private final Map<MetricType, Boolean> visible = new LinkedHashMap<>();
    private String title = "训练曲线";
    private String xLabel = "Epoch";
    private String yLabel = "值";

    public TrainingChart() {
        setPreferredSize(new Dimension(600, 400));
        setBackground(Color.WHITE);

        for (MetricType type : MetricType.values()) {
            data.put(type, new ArrayList<>());
            visible.put(type, true);
        }
    }

    public void addDataPoint(MetricType type, double value) {
        data.get(type).add(value);
        repaint();
    }

    public void clearData() {
        for (List<Double> list : data.values()) {
            list.clear();
        }
        repaint();
    }

    public void setMetricVisible(MetricType type, boolean vis) {
        visible.put(type, vis);
        repaint();
    }

    public void setTitle(String title) {
        this.title = title;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();
        int margin = 60;
        int plotW = w - 2 * margin;
        int plotH = h - 2 * margin;

        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, w, h);

        g2.setColor(Color.BLACK);
        g2.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        FontMetrics fm = g2.getFontMetrics();
        int titleWidth = fm.stringWidth(title);
        g2.drawString(title, (w - titleWidth) / 2, 25);

        g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        g2.drawString(xLabel, w / 2 - 20, h - 10);

        g2.setColor(Color.LIGHT_GRAY);
        g2.drawRect(margin, margin, plotW, plotH);

        for (int i = 0; i <= 5; i++) {
            int y = margin + (plotH * i) / 5;
            g2.setColor(Color.LIGHT_GRAY);
            g2.drawLine(margin, y, margin + plotW, y);
        }

        double globalMin = Double.MAX_VALUE;
        double globalMax = -Double.MAX_VALUE;
        int maxPoints = 0;

        for (Map.Entry<MetricType, List<Double>> entry : data.entrySet()) {
            if (!visible.get(entry.getKey())) continue;
            List<Double> values = entry.getValue();
            if (values.isEmpty()) continue;
            maxPoints = Math.max(maxPoints, values.size());
            for (double v : values) {
                if (!Double.isNaN(v)) {
                    globalMin = Math.min(globalMin, v);
                    globalMax = Math.max(globalMax, v);
                }
            }
        }

        if (maxPoints == 0) return;

        if (globalMin == globalMax) {
            globalMin -= 1;
            globalMax += 1;
        }

        double range = globalMax - globalMin;
        globalMin -= range * 0.05;
        globalMax += range * 0.05;

        g2.setColor(Color.BLACK);
        g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
        for (int i = 0; i <= 5; i++) {
            double val = globalMax - (globalMax - globalMin) * i / 5.0;
            String label = String.format("%.4f", val);
            int labelWidth = g2.getFontMetrics().stringWidth(label);
            g2.drawString(label, margin - labelWidth - 5, margin + (plotH * i) / 5 + 4);
        }

        for (int i = 0; i <= 5; i++) {
            int epoch = (maxPoints * i) / 5;
            String label = String.valueOf(epoch);
            int labelWidth = g2.getFontMetrics().stringWidth(label);
            g2.drawString(label, margin + (plotW * i) / 5 - labelWidth / 2, margin + plotH + 15);
        }

        for (Map.Entry<MetricType, List<Double>> entry : data.entrySet()) {
            MetricType type = entry.getKey();
            if (!visible.get(type)) continue;
            List<Double> values = entry.getValue();
            if (values.size() < 2) continue;

            g2.setColor(type.color);
            g2.setStroke(new BasicStroke(2f));

            Path2D path = new Path2D.Double();
            boolean first = true;
            for (int i = 0; i < values.size(); i++) {
                double val = values.get(i);
                if (Double.isNaN(val)) continue;

                double x = margin + (plotW * (double) i / (maxPoints - 1));
                double y = margin + plotH - (plotH * (val - globalMin) / (globalMax - globalMin));

                if (first) {
                    path.moveTo(x, y);
                    first = false;
                } else {
                    path.lineTo(x, y);
                }
            }
            g2.draw(path);
        }

        int legendX = margin + 10;
        int legendY = margin + 10;
        g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));

        for (MetricType type : MetricType.values()) {
            if (!visible.get(type)) continue;
            if (data.get(type).isEmpty()) continue;

            g2.setColor(type.color);
            g2.fillRect(legendX, legendY, 12, 12);
            g2.setColor(Color.BLACK);
            g2.drawString(type.label, legendX + 16, legendY + 10);
            legendY += 18;
        }
    }

    public JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        panel.setBorder(BorderFactory.createTitledBorder("显示选项"));

        for (MetricType type : MetricType.values()) {
            JCheckBox cb = new JCheckBox(type.label, visible.get(type));
            cb.addActionListener(e -> setMetricVisible(type, cb.isSelected()));
            panel.add(cb);
        }

        JButton clearBtn = new JButton("清除");
        clearBtn.addActionListener(e -> clearData());
        panel.add(clearBtn);

        return panel;
    }
}

package machain.ui;

import javax.swing.*;
import java.awt.*;

public class ConfusionMatrixPanel extends JPanel {

    private int[][] matrix;
    private String[] classNames;
    private String title = "混淆矩阵";

    public ConfusionMatrixPanel() {
        setPreferredSize(new Dimension(400, 400));
        setBackground(Color.WHITE);
    }

    public void setMatrix(int[][] matrix, String[] classNames) {
        this.matrix = matrix;
        this.classNames = classNames;
        repaint();
    }

    public void clear() {
        matrix = null;
        classNames = null;
        repaint();
    }

    public void setTitle(String title) {
        this.title = title;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (matrix == null || classNames == null) {
            g.setColor(Color.GRAY);
            g.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 14));
            g.drawString("暂无数据", getWidth() / 2 - 40, getHeight() / 2);
            return;
        }

        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();
        int margin = 80;
        int size = Math.min(w, h) - 2 * margin;
        int cellSize = size / matrix.length;

        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, w, h);

        g2.setColor(Color.BLACK);
        g2.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        FontMetrics fm = g2.getFontMetrics();
        int titleWidth = fm.stringWidth(title);
        g2.drawString(title, (w - titleWidth) / 2, 25);

        int maxVal = 0;
        for (int[] row : matrix) {
            for (int val : row) {
                maxVal = Math.max(maxVal, val);
            }
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                int x = margin + j * cellSize;
                int y = margin + i * cellSize;
                int val = matrix[i][j];

                float intensity = maxVal > 0 ? (float) val / maxVal : 0;
                if (i == j) {
                    g2.setColor(new Color(0, (int)(200 * intensity), 0));
                } else {
                    g2.setColor(new Color((int)(200 * intensity), 0, 0));
                }
                g2.fillRect(x, y, cellSize, cellSize);

                g2.setColor(Color.BLACK);
                g2.drawRect(x, y, cellSize, cellSize);

                String text = String.valueOf(val);
                int textWidth = g2.getFontMetrics().stringWidth(text);
                g2.drawString(text, x + (cellSize - textWidth) / 2, y + cellSize / 2 + 5);
            }
        }

        g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        for (int i = 0; i < classNames.length; i++) {
            String name = classNames[i];
            if (name.length() > 8) name = name.substring(0, 7) + "..";

            int textWidth = g2.getFontMetrics().stringWidth(name);
            g2.drawString(name, margin - textWidth - 5, margin + i * cellSize + cellSize / 2 + 4);

            g2.drawString(name, margin + i * cellSize + cellSize / 2 - textWidth / 2, margin + matrix.length * cellSize + 15);
        }

        g2.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g2.drawString("预测", margin + size / 2 - 15, h - 10);

        g2.rotate(-Math.PI / 2);
        g2.drawString("实际", -(margin + size / 2 + 20), 20);
    }

    public static int[][] computeConfusionMatrix(double[][] predictions, double[][] targets) {
        int numClasses = targets[0].length;
        int[][] matrix = new int[numClasses][numClasses];

        for (int i = 0; i < predictions.length; i++) {
            int predClass = 0;
            int trueClass = 0;

            for (int j = 1; j < numClasses; j++) {
                if (predictions[i][j] > predictions[i][predClass]) predClass = j;
                if (targets[i][j] > targets[i][trueClass]) trueClass = j;
            }

            matrix[trueClass][predClass]++;
        }

        return matrix;
    }

    public static double computeAccuracy(int[][] matrix) {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                total += matrix[i][j];
                if (i == j) correct += matrix[i][j];
            }
        }
        return total > 0 ? (double) correct / total : 0;
    }
}

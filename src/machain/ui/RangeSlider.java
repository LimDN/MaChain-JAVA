package machain.ui;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.ArrayList;
import java.util.List;

public class RangeSlider extends JPanel {

    public interface RangeListener {
        void rangeChanged(double low, double high);
    }

    private static final int THUMB_R = 7;
    private static final int TRACK_H = 6;

    private double minVal, maxVal;
    private double lowVal, highVal;
    private final List<RangeListener> listeners = new ArrayList<>();

    private enum DragTarget { NONE, LOW, HIGH }
    private DragTarget dragging = DragTarget.NONE;

    public RangeSlider(double min, double max, double low, double high) {
        this.minVal = min;
        this.maxVal = max;
        this.lowVal = low;
        this.highVal = high;
        setPreferredSize(new Dimension(260, 36));

        MouseAdapter clickHandler = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int x = e.getX();
                int lowX = valToX(lowVal);
                int highX = valToX(highVal);
                if (Math.abs(x - lowX) < THUMB_R + 3) dragging = DragTarget.LOW;
                else if (Math.abs(x - highX) < THUMB_R + 3) dragging = DragTarget.HIGH;
                else {
                    double clickVal = xToVal(x);
                    if (Math.abs(clickVal - lowVal) < Math.abs(clickVal - highVal)) {
                        lowVal = clamp(clickVal, minVal, highVal);
                        dragging = DragTarget.LOW;
                    } else {
                        highVal = clamp(clickVal, lowVal, maxVal);
                        dragging = DragTarget.HIGH;
                    }
                    repaint();
                    fireRangeChanged();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                dragging = DragTarget.NONE;
            }
        };
        addMouseListener(clickHandler);
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (dragging == DragTarget.NONE) return;
                double v = clamp(xToVal(e.getX()), minVal, maxVal);
                if (dragging == DragTarget.LOW) {
                    lowVal = Math.min(v, highVal);
                } else {
                    highVal = Math.max(v, lowVal);
                }
                repaint();
                fireRangeChanged();
            }
        });
    }

    public void addRangeListener(RangeListener l) { listeners.add(l); }
    private void fireRangeChanged() {
        for (RangeListener l : listeners) l.rangeChanged(lowVal, highVal);
    }

    public double getLow() { return lowVal; }
    public double getHigh() { return highVal; }

    public void setRange(double low, double high) {
        this.lowVal = clamp(low, minVal, maxVal);
        this.highVal = clamp(high, minVal, maxVal);
        if (this.lowVal > this.highVal) { double t = this.lowVal; this.lowVal = this.highVal; this.highVal = t; }
        repaint();
        fireRangeChanged();
    }

    private double clamp(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }

    private int valToX(double v) {
        int w = getWidth() - 2 * THUMB_R;
        if (w <= 0) w = 200;
        return THUMB_R + (int) ((v - minVal) / (maxVal - minVal) * w);
    }

    private double xToVal(int x) {
        int w = getWidth() - 2 * THUMB_R;
        if (w <= 0) w = 200;
        double ratio = (double)(x - THUMB_R) / w;
        ratio = clamp(ratio, 0, 1);
        return minVal + ratio * (maxVal - minVal);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int y = getHeight() / 2;
        int x1 = THUMB_R, x2 = getWidth() - THUMB_R;

        g2.setColor(new Color(220, 220, 220));
        g2.setStroke(new BasicStroke(TRACK_H));
        g2.drawLine(x1, y, x2, y);

        int lowX = valToX(lowVal);
        int highX = valToX(highVal);

        g2.setColor(new Color(255, 200, 200, 80));
        g2.fillRect(lowX, y - TRACK_H, highX - lowX, TRACK_H * 2);

        g2.setColor(new Color(180, 0, 0));
        g2.setStroke(new BasicStroke(2));
        g2.drawLine(lowX, y - TRACK_H, lowX, y + TRACK_H);
        g2.drawLine(highX, y - TRACK_H, highX, y + TRACK_H);

        g2.setColor(new Color(200, 40, 40));
        g2.fillOval(lowX - THUMB_R, y - THUMB_R, THUMB_R * 2, THUMB_R * 2);
        g2.fillOval(highX - THUMB_R, y - THUMB_R, THUMB_R * 2, THUMB_R * 2);

        g2.setColor(Color.WHITE);
        g2.fillOval(lowX - 3, y - 3, 6, 6);
        g2.fillOval(highX - 3, y - 3, 6, 6);

        g2.dispose();
    }
}

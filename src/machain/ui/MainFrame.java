package machain.ui;

import machain.api.ApiServer;
import machain.config.Config;
import machain.core.ConsoleTrainingCallback;
import machain.core.Trainer;
import machain.core.DualTrainer;
import machain.core.TrainingCallback;
import machain.core.TrainingResult;
import machain.data.DataSet;
import machain.data.TextProcessor;
import machain.data.SampleExtractor;
import machain.gpu.GPUAccelerator;
import machain.io.SaveManager;
import machain.io.SampleScraper;
import machain.io.OutputExporter;
import machain.generation.TextGenerator;
import machain.network.Network;
import machain.network.Network.Activation;
import machain.network.OutputMode;
import machain.network.DualStreamNetwork;
import machain.output.OutputManager;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class MainFrame extends JFrame {

    private Network network;
    private DualStreamNetwork dualNetwork;
    private DataSet dataSet;
    private DataSet trainSet, testSet;
    private TextProcessor textProcessor;

    private DataSet structDataSet, structTrainSet, structTestSet;
    private DataSet textDataSet, textTrainSet, textTestSet;
    private TextProcessor structTextProcessor;

    private OutputMode outputMode = OutputMode.REGRESSION;
    private boolean dualMode = false;
    private final SaveManager saveManager;
    private final OutputManager outputManager;
    private final Config config;
    private Trainer trainer;
    private DualTrainer dualTrainer;
    private TrainingResult lastResult;
    private Thread trainThread;

    private JTextField layerField;
    private JComboBox<String> activationBox;
    private JTextField lrField, momentumField;
    private JComboBox<OutputMode> outputModeBox;

    private JSlider complexitySlider;
    private JTextField complexityField;
    private JSlider precisionSlider;
    private JTextField precisionField;
    private JSlider regularizationSlider;
    private JTextField regularizationField;
    private JCheckBox gpuCheckBox;
    private JLabel gpuStatusLabel;

    private JTextField structLayerField, textLayerField;
    private JTextField bridgeDimField, crossWeightField;
    private JLabel structDataLabel, textDataLabel;

    private JTextArea dataAnalysisArea;
    private JTextArea dataPreviewArea;
    private JLabel dataStatusLabel;

    private JTextField epochField;
    private JTextField logIntervalField;
    private JProgressBar progressBar;
    private JTextArea trainLogArea;
    private JButton startBtn, stopBtn;
    private JLabel lossLabel, epochLabel, accuracyLabel;
    private JLabel crossLossLabel;
    private JLabel modeToggleLabel;
    private TrainingChart trainingChart;
    private ConfusionMatrixPanel confusionMatrixPanel;

    private JTextField weightDecayField;
    private JTextField dropoutField;
    private JTextField batchSizeField;
    private JComboBox<String> lrScheduleBox;
    private JTextField earlyStopField;
    private JSlider learnSpecSlider;
    private JLabel learnSpecValueLabel;
    private double learnSpecCoeff = 1.0;

    private Network appNetwork;
    private TextProcessor appTextProcessor;
    private JLabel appModelInfoLabel;
    private JTextArea appInputArea;
    private JTextArea appOutputArea;
    private JTextArea appThinkingArea;
    private JTextArea appHistoryArea;
    private JLabel appStatusLabel;
    private OutputMode appOutputMode = OutputMode.REGRESSION;
    private String[] appLabelNames;
    private TextGenerator textGenerator;
    private machain.network.EnhancedNetwork enhancedNetwork;
    private JCheckBox useEnhancedModeBox;
    private List<String> originalTexts = new ArrayList<>();
    private JCheckBox preservePunctuationBox;
    private machain.api.ApiServer apiServer;
    private JLabel apiStatusLabel;
    private JTextField apiPortField;
    private JButton apiToggleBtn;

    private SampleScraper sampleScraper;
    private JTextArea interfaceLogArea;
    private JLabel clipboardStatusLabel;
    private JLabel folderWatchStatusLabel;
    private JTextField urlInputField;
    private JTextArea scrapedTextArea;

    private RangeSlider biasRangeSlider;
    private JLabel biasRangeValueLabel;
    private JComboBox<String> biasWarningBox;
    private JLabel biasStatusLabel;
    private double biasLow = 0.0, biasHigh = 0.0;
    private enum BiasAction { NONE, WARN, ABORT }
    private BiasAction biasAction = BiasAction.NONE;

    private DefaultTableModel saveTableModel;
    private JTable saveTable;

    private JComboBox<String> outputFormatBox;
    private JTextArea outputLogArea;

    private JTextField applyInputField;
    private JTextArea applyOutputArea;
    private JTextArea applyThinkingArea;
    private JButton applyBtn, discardBtn, keepBtn;
    private JLabel applyStatusLabel;

    private List<double[]> keptInputs = new ArrayList<>();
    private List<double[]> keptOutputs = new ArrayList<>();
    private List<Double> keptWeights = new ArrayList<>();

    public MainFrame() {
        super("MaChain v1.5 - 轻量级神经网络训练器");
        String baseDir = System.getProperty("user.dir") + File.separator + "MaChain";
        saveManager = new SaveManager(baseDir);
        outputManager = new OutputManager(baseDir);
        config = new Config();
        config.setSaveDir(saveManager.getSaveDir());
        config.setOutputDir(outputManager.getOutputDir());
        loadConfigSilently();
        try {
            GPUAccelerator.getInstance().initialize();
        } catch (Throwable t) {
            System.err.println("[启动] GPU初始化跳过: " + t.getMessage());
        }
        initUI();
        setTitle("MaChain-轻量化活字乱输器平台 v1.0");
        setSize(1100, 800);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private void loadConfigSilently() {
        File f = new File("machain.properties");
        if (f.exists()) {
            try { config.loadFromFile(f.getAbsolutePath()); } catch (Exception ignored) {}
        }
    }

    private void initUI() {
        JPanel topBar = buildApplyToolbar();
        add(topBar, BorderLayout.NORTH);

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("网络配置", buildConfigPanel());
        tabs.addTab("数据管理", buildDataPanel());
        tabs.addTab("训练", buildTrainPanel());
        tabs.addTab("可视化", buildVisualizationPanel());
        tabs.addTab("输出", buildOutputPanel());
        tabs.addTab("实例应用", buildAppPanel());
        tabs.addTab("接口中心", buildInterfacePanel());
        add(tabs, BorderLayout.CENTER);

        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.setBorder(BorderFactory.createEtchedBorder());

        JPanel statusBar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        statusBar.add(new JLabel("MaChain v1.5 | 精密度/复杂度可调 | 偏误监测 | 单流/双流切换 | 模糊交叉蒸馏"));
        bottomPanel.add(statusBar, BorderLayout.CENTER);

        JPanel saveBtnPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 6, 4));
        JButton saveProgressBtn = new JButton("保存进度");
        saveProgressBtn.addActionListener(e -> saveProgress());
        saveBtnPanel.add(saveProgressBtn);
        JButton saveSkillBtn = new JButton("保存Skill");
        saveSkillBtn.addActionListener(e -> saveSkill());
        saveBtnPanel.add(saveSkillBtn);
        JButton loadBtn = new JButton("加载存档");
        loadBtn.addActionListener(e -> loadSaveDialog());
        saveBtnPanel.add(loadBtn);
        JButton saveVocabBtn = new JButton("保存词库");
        saveVocabBtn.addActionListener(e -> saveVocabulary());
        saveBtnPanel.add(saveVocabBtn);
        JButton loadVocabBtn = new JButton("加载词库");
        loadVocabBtn.addActionListener(e -> loadVocabulary());
        saveBtnPanel.add(loadVocabBtn);
        bottomPanel.add(saveBtnPanel, BorderLayout.EAST);

        add(bottomPanel, BorderLayout.SOUTH);
    }

    // ========== 应用工具栏 ==========
    private JPanel buildApplyToolbar() {
        JPanel toolbar = new JPanel(new BorderLayout(4, 0));
        toolbar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 1, 0, Color.GRAY),
                BorderFactory.createEmptyBorder(4, 6, 4, 6)));

        JPanel inputPanel = new JPanel(new BorderLayout(3, 0));
        inputPanel.add(new JLabel("输入:"), BorderLayout.WEST);
        applyInputField = new JTextField(20);
        inputPanel.add(applyInputField, BorderLayout.CENTER);
        toolbar.add(inputPanel, BorderLayout.WEST);

        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        applyBtn = new JButton("应用");
        applyBtn.setFont(applyBtn.getFont().deriveFont(Font.BOLD));
        applyBtn.addActionListener(e -> applyInput());
        btnPanel.add(applyBtn);

        discardBtn = new JButton("丢弃");
        discardBtn.setEnabled(false);
        discardBtn.setForeground(new Color(180, 0, 0));
        discardBtn.addActionListener(e -> discardOutput());
        btnPanel.add(discardBtn);

        keepBtn = new JButton("保留");
        keepBtn.setEnabled(false);
        keepBtn.setForeground(new Color(0, 120, 0));
        keepBtn.addActionListener(e -> keepOutput());
        btnPanel.add(keepBtn);

        applyStatusLabel = new JLabel(" ");
        applyStatusLabel.setForeground(Color.GRAY);
        btnPanel.add(applyStatusLabel);

        toolbar.add(btnPanel, BorderLayout.CENTER);

        JPanel displayPanel = new JPanel(new GridLayout(1, 2, 4, 0));
        applyOutputArea = new JTextArea(2, 15);
        applyOutputArea.setEditable(false);
        applyOutputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        applyOutputArea.setLineWrap(true);
        JScrollPane outScroll = new JScrollPane(applyOutputArea);
        outScroll.setBorder(BorderFactory.createTitledBorder("输出结果"));
        outScroll.setPreferredSize(new Dimension(200, 50));
        displayPanel.add(outScroll);

        applyThinkingArea = new JTextArea(2, 15);
        applyThinkingArea.setEditable(false);
        applyThinkingArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        applyThinkingArea.setLineWrap(true);
        JScrollPane thinkScroll = new JScrollPane(applyThinkingArea);
        thinkScroll.setBorder(BorderFactory.createTitledBorder("思考过程"));
        thinkScroll.setPreferredSize(new Dimension(200, 50));
        displayPanel.add(thinkScroll);

        toolbar.add(displayPanel, BorderLayout.EAST);

        return toolbar;
    }

    private void applyInput() {
        String text = applyInputField.getText().trim();
        if (text.isEmpty()) {
            applyStatusLabel.setText("请输入数据");
            return;
        }

        if (dualMode && dualNetwork != null) {
            applyDualInput(text);
        } else if (network != null) {
            applySingleInput(text);
        } else {
            applyStatusLabel.setText("请先构建网络");
        }
    }

    private void applySingleInput(String text) {
        try {
            double[] input;
            if (textProcessor != null) {
                input = textProcessor.vectorize(text);
            } else {
                String[] parts = text.split(",");
                input = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    input[i] = Double.parseDouble(parts[i].trim());
                }
            }

            if (input.length != network.getLayerSizes()[0]) {
                applyStatusLabel.setText("输入维度不匹配 (期望" + network.getLayerSizes()[0] + ")");
                return;
            }

            StringBuilder thinking = new StringBuilder();
            thinking.append("网络: ").append(network.getArchitecture()).append("\n");
            thinking.append("输入维度: ").append(input.length).append("\n");
            thinking.append("输出模式: ").append(outputMode.getDisplayName()).append("\n");

            double[] output = network.forward(input);

            thinking.append("激活函数: ").append(network.getLayerActivations()[0]).append("\n");
            thinking.append("参数量: ").append(network.countParameters());

            StringBuilder outputStr = new StringBuilder();
            if (outputMode == OutputMode.REGRESSION) {
                for (int i = 0; i < output.length; i++) {
                    if (i > 0) outputStr.append(", ");
                    outputStr.append(String.format("%.6f", output[i]));
                }
            } else {
                int predClass = outputMode.predictClass(output);
                outputStr.append("类别: ").append(predClass).append("\n");
                for (int i = 0; i < output.length; i++) {
                    outputStr.append(String.format("  P(%d)=%.4f", i, output[i]));
                    if (i < output.length - 1) outputStr.append("\n");
                }
                thinking.append("\n预测: 类别 ").append(predClass);
            }

            applyOutputArea.setText(outputStr.toString());
            applyThinkingArea.setText(thinking.toString());

            discardBtn.setEnabled(true);
            keepBtn.setEnabled(true);
            applyStatusLabel.setText("已完成推理 | 可丢弃或保留");

        } catch (Exception ex) {
            applyOutputArea.setText("错误");
            applyThinkingArea.setText("解析失败: " + ex.getMessage());
            applyStatusLabel.setText("输入解析失败");
        }
    }

    private void applyDualInput(String text) {
        try {
            StringBuilder thinking = new StringBuilder();
            thinking.append("双流网络推理\n");
            thinking.append("桥接维度: ").append(dualNetwork.getBridge().getBridgeDim()).append("\n");
            thinking.append("交叉权重: ").append(dualNetwork.getCrossLossWeight()).append("\n");

            double[] structInput;
            double[] textInput;

            if (text.contains("|||")) {
                String[] halves = text.split("\\|\\|\\|", 2);
                String[] sParts = halves[0].trim().split(",");
                structInput = new double[sParts.length];
                for (int i = 0; i < sParts.length; i++) structInput[i] = Double.parseDouble(sParts[i].trim());

                if (structTextProcessor != null) {
                    textInput = structTextProcessor.vectorize(halves[1].trim());
                } else {
                    textInput = new double[dualNetwork.getTextNetwork().getLayerSizes()[0]];
                    Arrays.fill(textInput, 0.01);
                }
            } else {
                String[] parts = text.split(",");
                int structDim = dualNetwork.getStructNetwork().getLayerSizes()[0];
                int textDim = dualNetwork.getTextNetwork().getLayerSizes()[0];

                structInput = new double[structDim];
                for (int i = 0; i < Math.min(parts.length, structDim); i++) {
                    structInput[i] = Double.parseDouble(parts[i].trim());
                }

                textInput = new double[textDim];
                Arrays.fill(textInput, 0.01);
            }

            thinking.append("结构化输入维度: ").append(structInput.length).append("\n");
            thinking.append("文本输入维度: ").append(textInput.length).append("\n");

            double[] structHidden = dualNetwork.extractStructHidden(structInput);
            double[] textHidden = dualNetwork.extractTextHidden(textInput);
            thinking.append("结构化隐表示: [");
            for (int i = 0; i < Math.min(structHidden.length, 4); i++) {
                thinking.append(String.format("%.3f", structHidden[i]));
                if (i < Math.min(structHidden.length, 4) - 1) thinking.append(",");
            }
            thinking.append(structHidden.length > 4 ? "...]\n" : "]\n");

            thinking.append("文本隐表示: [");
            for (int i = 0; i < Math.min(textHidden.length, 4); i++) {
                thinking.append(String.format("%.3f", textHidden[i]));
                if (i < Math.min(textHidden.length, 4) - 1) thinking.append(",");
            }
            thinking.append(textHidden.length > 4 ? "...]\n" : "]\n");

            double[] membership = dualNetwork.getBridge().getMembershipWeights();
            thinking.append(String.format("隶属度: [%.3f, %.3f]", membership[0], membership[1]));

            double[] output = dualNetwork.fusedPredict(structInput, textInput);

            StringBuilder outputStr = new StringBuilder();
            for (int i = 0; i < output.length; i++) {
                if (i > 0) outputStr.append(", ");
                outputStr.append(String.format("%.6f", output[i]));
            }

            applyOutputArea.setText(outputStr.toString());
            applyThinkingArea.setText(thinking.toString());

            discardBtn.setEnabled(true);
            keepBtn.setEnabled(true);
            applyStatusLabel.setText("双流推理完成 | 可丢弃或保留");

        } catch (Exception ex) {
            applyOutputArea.setText("错误");
            applyThinkingArea.setText("双流推理失败: " + ex.getMessage());
            applyStatusLabel.setText("输入解析失败");
        }
    }

    private void discardOutput() {
        applyOutputArea.setText("");
        applyThinkingArea.setText("");
        applyInputField.setText("");
        discardBtn.setEnabled(false);
        keepBtn.setEnabled(false);
        applyStatusLabel.setText("已丢弃");
    }

    private void keepOutput() {
        try {
            String text = applyInputField.getText().trim();
            double[] input;
            if (textProcessor != null) {
                input = textProcessor.vectorize(text);
            } else if (text.contains("|||")) {
                String[] halves = text.split("\\|\\|\\|", 2);
                String[] sParts = halves[0].trim().split(",");
                input = new double[sParts.length];
                for (int i = 0; i < sParts.length; i++) input[i] = Double.parseDouble(sParts[i].trim());
            } else {
                String[] parts = text.split(",");
                input = new double[parts.length];
                for (int i = 0; i < parts.length; i++) input[i] = Double.parseDouble(parts[i].trim());
            }

            double[] output;
            if (dualMode && dualNetwork != null) {
                double[] structInput, textInput;
                if (text.contains("|||")) {
                    String[] halves = text.split("\\|\\|\\|", 2);
                    String[] sParts = halves[0].trim().split(",");
                    structInput = new double[sParts.length];
                    for (int i = 0; i < sParts.length; i++) structInput[i] = Double.parseDouble(sParts[i].trim());
                    if (structTextProcessor != null) {
                        textInput = structTextProcessor.vectorize(halves[1].trim());
                    } else {
                        textInput = new double[dualNetwork.getTextNetwork().getLayerSizes()[0]];
                        Arrays.fill(textInput, 0.01);
                    }
                } else {
                    int structDim = dualNetwork.getStructNetwork().getLayerSizes()[0];
                    int textDim = dualNetwork.getTextNetwork().getLayerSizes()[0];
                    String[] parts = text.split(",");
                    structInput = new double[structDim];
                    for (int i = 0; i < Math.min(parts.length, structDim); i++)
                        structInput[i] = Double.parseDouble(parts[i].trim());
                    textInput = new double[textDim];
                    Arrays.fill(textInput, 0.01);
                }
                output = dualNetwork.fusedPredict(structInput, textInput);
            } else {
                output = network.forward(input);
            }

            keptInputs.add(input);
            keptOutputs.add(output);
            keptWeights.add(2.0);

            applyStatusLabel.setText("已保留 (权重x2.0) | 累计: " + keptInputs.size() + " 条");
            discardOutput();

        } catch (Exception ex) {
            applyStatusLabel.setText("保留失败: " + ex.getMessage());
        }
    }

    // ========== 网络配置面板 (合并单流+双流) ==========
    private JPanel buildConfigPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel scrollContent = new JPanel();
        scrollContent.setLayout(new BoxLayout(scrollContent, BoxLayout.Y_AXIS));

        // 公共参数
        JPanel commonPanel = new JPanel(new GridBagLayout());
        commonPanel.setBorder(BorderFactory.createTitledBorder("公共参数"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 8, 4, 8);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        int row = 0;
        gbc.gridx = 0; gbc.gridy = row;
        commonPanel.add(new JLabel("激活函数:"), gbc);
        gbc.gridx = 1;
        activationBox = new JComboBox<>(new String[]{"SIGMOID", "RELU", "TANH"});
        activationBox.setSelectedItem(config.getActivation());
        commonPanel.add(activationBox, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        commonPanel.add(new JLabel("输出模式:"), gbc);
        gbc.gridx = 1;
        outputModeBox = new JComboBox<>(OutputMode.values());
        outputModeBox.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                if (value instanceof OutputMode mode) {
                    setText(mode.getDisplayName() + " - " + mode.getDescription());
                }
                return this;
            }
        });
        commonPanel.add(outputModeBox, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        commonPanel.add(new JLabel("学习率:"), gbc);
        gbc.gridx = 1;
        lrField = new JTextField(String.valueOf(config.getLearningRate()), 10);
        commonPanel.add(lrField, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        commonPanel.add(new JLabel("动量:"), gbc);
        gbc.gridx = 1;
        momentumField = new JTextField(String.valueOf(config.getMomentum()), 10);
        commonPanel.add(momentumField, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        JPanel cfgBtnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        JButton saveConfigBtn = new JButton("保存配置");
        saveConfigBtn.addActionListener(e -> saveConfig());
        cfgBtnRow.add(saveConfigBtn);
        JButton loadConfigBtn = new JButton("加载配置");
        loadConfigBtn.addActionListener(e -> loadConfig());
        cfgBtnRow.add(loadConfigBtn);
        commonPanel.add(cfgBtnRow, gbc);

        scrollContent.add(commonPanel);
        scrollContent.add(Box.createVerticalStrut(6));

        // 精密度/复杂度面板
        JPanel modelPanel = new JPanel(new GridBagLayout());
        modelPanel.setBorder(BorderFactory.createTitledBorder("模型精密度 / 复杂度"));
        GridBagConstraints mgbc = new GridBagConstraints();
        mgbc.insets = new Insets(4, 8, 4, 8);
        mgbc.anchor = GridBagConstraints.WEST;
        mgbc.fill = GridBagConstraints.HORIZONTAL;

        int mrow = 0;
        mgbc.gridx = 0; mgbc.gridy = mrow;
        modelPanel.add(new JLabel("复杂度:"), mgbc);
        mgbc.gridx = 1;
        complexitySlider = new JSlider(1, 200, 10);
        complexitySlider.setPreferredSize(new Dimension(180, 24));
        complexitySlider.setMajorTickSpacing(20);
        complexitySlider.setMinorTickSpacing(5);
        complexitySlider.setPaintTicks(true);
        modelPanel.add(complexitySlider, mgbc);
        mgbc.gridx = 2;
        complexityField = new JTextField("10", 5);
        modelPanel.add(complexityField, mgbc);
        mgbc.gridx = 3;
        JLabel complexityHint = new JLabel("(基数=复杂度x8, 最大1600)");
        complexityHint.setForeground(Color.GRAY);
        complexityHint.setFont(complexityHint.getFont().deriveFont(11f));
        modelPanel.add(complexityHint, mgbc);

        mrow++;
        mgbc.gridx = 0; mgbc.gridy = mrow;
        modelPanel.add(new JLabel("精密度:"), mgbc);
        mgbc.gridx = 1;
        precisionSlider = new JSlider(1, 20, 5);
        precisionSlider.setPreferredSize(new Dimension(180, 24));
        precisionSlider.setMajorTickSpacing(5);
        precisionSlider.setPaintTicks(true);
        modelPanel.add(precisionSlider, mgbc);
        mgbc.gridx = 2;
        precisionField = new JTextField("5", 4);
        modelPanel.add(precisionField, mgbc);
        mgbc.gridx = 3;
        JLabel precisionHint = new JLabel("(lr=0.1/1.8^(p-1), 越高越细)");
        precisionHint.setForeground(Color.GRAY);
        precisionHint.setFont(precisionHint.getFont().deriveFont(11f));
        modelPanel.add(precisionHint, mgbc);

        mrow++;
        mgbc.gridx = 0; mgbc.gridy = mrow;
        modelPanel.add(new JLabel("正则化:"), mgbc);
        mgbc.gridx = 1;
        regularizationSlider = new JSlider(0, 10, 0);
        regularizationSlider.setPreferredSize(new Dimension(180, 24));
        regularizationSlider.setMajorTickSpacing(2);
        regularizationSlider.setPaintTicks(true);
        modelPanel.add(regularizationSlider, mgbc);
        mgbc.gridx = 2;
        regularizationField = new JTextField("0.0", 4);
        modelPanel.add(regularizationField, mgbc);
        mgbc.gridx = 3;
        JLabel regHint = new JLabel("(L2正则化强度)");
        regHint.setForeground(Color.GRAY);
        regHint.setFont(regHint.getFont().deriveFont(11f));
        modelPanel.add(regHint, mgbc);

        mrow++;
        mgbc.gridx = 0; mgbc.gridy = mrow;
        modelPanel.add(new JLabel("GPU加速:"), mgbc);
        mgbc.gridx = 1; mgbc.gridwidth = 2;
        gpuCheckBox = new JCheckBox("启用 GPU 加速 (OpenCL)");
        gpuCheckBox.addActionListener(e -> toggleGPU());
        modelPanel.add(gpuCheckBox, mgbc);
        mgbc.gridwidth = 1;

        mrow++;
        mgbc.gridx = 0; mgbc.gridy = mrow; mgbc.gridwidth = 4;
        gpuStatusLabel = new JLabel("GPU状态: " + GPUAccelerator.getInstance().getStatus());
        gpuStatusLabel.setForeground(Color.GRAY);
        gpuStatusLabel.setFont(gpuStatusLabel.getFont().deriveFont(11f));
        modelPanel.add(gpuStatusLabel, mgbc);
        mgbc.gridwidth = 1;

        mrow++;
        mgbc.gridx = 0; mgbc.gridy = mrow; mgbc.gridwidth = 4;
        JPanel paramBtnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        JButton applyParamsBtn = new JButton("应用参数");
        applyParamsBtn.addActionListener(e -> applyModelParams());
        paramBtnRow.add(applyParamsBtn);
        JButton resetParamsBtn = new JButton("重置默认");
        resetParamsBtn.addActionListener(e -> resetModelParams());
        paramBtnRow.add(resetParamsBtn);
        modelPanel.add(paramBtnRow, mgbc);
        mgbc.gridwidth = 1;

        complexitySlider.addChangeListener(e -> {
            if (!complexitySlider.getValueIsAdjusting())
                complexityField.setText(String.valueOf(complexitySlider.getValue()));
        });
        precisionSlider.addChangeListener(e -> {
            if (!precisionSlider.getValueIsAdjusting())
                precisionField.setText(String.valueOf(precisionSlider.getValue()));
        });
        regularizationSlider.addChangeListener(e -> {
            if (!regularizationSlider.getValueIsAdjusting())
                regularizationField.setText(String.format("%.1f", regularizationSlider.getValue() * 0.001));
        });

        scrollContent.add(modelPanel);
        scrollContent.add(Box.createVerticalStrut(6));

        // 单流配置
        JPanel singlePanel = new JPanel(new GridBagLayout());
        singlePanel.setBorder(BorderFactory.createTitledBorder("单流网络"));
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 8, 4, 8);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        row = 0;
        gbc.gridx = 0; gbc.gridy = row;
        singlePanel.add(new JLabel("网络结构 (逗号分隔):"), gbc);
        gbc.gridx = 1; gbc.gridwidth = 2;
        layerField = new JTextField(arrayToString(config.getLayerSizes()), 20);
        singlePanel.add(layerField, gbc);
        gbc.gridwidth = 1;

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        JPanel singleBtnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        JButton buildBtn = new JButton("构建单流网络");
        buildBtn.addActionListener(e -> buildNetwork());
        singleBtnRow.add(buildBtn);
        singlePanel.add(singleBtnRow, gbc);

        scrollContent.add(singlePanel);
        scrollContent.add(Box.createVerticalStrut(6));

        // 双流配置
        JPanel dualPanel = new JPanel(new GridBagLayout());
        dualPanel.setBorder(BorderFactory.createTitledBorder("双流网络 (结构化 + 文本)"));
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 8, 4, 8);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        row = 0;
        gbc.gridx = 0; gbc.gridy = row;
        dualPanel.add(new JLabel("结构化子网络:"), gbc);
        gbc.gridx = 1; gbc.gridwidth = 2;
        structLayerField = new JTextField("2,8,1", 20);
        dualPanel.add(structLayerField, gbc);
        gbc.gridwidth = 1;

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        dualPanel.add(new JLabel("文本子网络:"), gbc);
        gbc.gridx = 1; gbc.gridwidth = 2;
        textLayerField = new JTextField("100,32,1", 20);
        dualPanel.add(textLayerField, gbc);
        gbc.gridwidth = 1;

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        dualPanel.add(new JLabel("桥接维度:"), gbc);
        gbc.gridx = 1;
        bridgeDimField = new JTextField("16", 8);
        dualPanel.add(bridgeDimField, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        dualPanel.add(new JLabel("交叉蒸馏权重:"), gbc);
        gbc.gridx = 1;
        crossWeightField = new JTextField("0.1", 8);
        dualPanel.add(crossWeightField, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        structDataLabel = new JLabel("结构化数据: 未加载");
        structDataLabel.setForeground(new Color(0, 100, 0));
        dualPanel.add(structDataLabel, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        textDataLabel = new JLabel("文本数据: 未加载");
        textDataLabel.setForeground(new Color(0, 0, 150));
        dualPanel.add(textDataLabel, gbc);

        row++;
        gbc.gridx = 0; gbc.gridy = row;
        JPanel dualBtnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        JButton buildDualBtn = new JButton("构建双流网络");
        buildDualBtn.addActionListener(e -> buildDualNetwork());
        dualBtnRow.add(buildDualBtn);
        JButton switchSingleBtn = new JButton("切回单流模式");
        switchSingleBtn.addActionListener(e -> {
            dualMode = false;
            if (modeToggleLabel != null) modeToggleLabel.setText("<html><u>模式: 单流</u></html>");
            JOptionPane.showMessageDialog(this, "已切换到单流模式");
        });
        dualBtnRow.add(switchSingleBtn);
        dualPanel.add(dualBtnRow, gbc);

        scrollContent.add(dualPanel);
        scrollContent.add(Box.createVerticalStrut(6));

        // 说明
        JTextArea infoArea = new JTextArea(6, 50);
        infoArea.setEditable(false);
        infoArea.setLineWrap(true);
        infoArea.setText("说明:\n" +
                "  单流: 输入层,隐藏层,...,输出层 的神经元数量\n" +
                "  双流: 结构化(数值CSV) + 文本(文本CSV) 两个子网络, 通过模糊桥接交叉蒸馏\n" +
                "  应用栏: 顶部输入数据后点'应用'推理, '保留'将输出加入训练集(权重x2.0)\n" +
                "  双流应用输入格式: 数值部分|||文本部分");
        scrollContent.add(infoArea);

        scrollContent.add(Box.createVerticalGlue());

        JScrollPane scrollPane = new JScrollPane(scrollContent);
        scrollPane.setBorder(null);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        panel.add(scrollPane, BorderLayout.CENTER);

        return panel;
    }

    private void buildNetwork() {
        try {
            String[] parts = layerField.getText().trim().split(",");
            int[] sizes = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                sizes[i] = Integer.parseInt(parts[i].trim());
                if (sizes[i] <= 0) throw new NumberFormatException("层大小须为正整数");
            }

            if (textProcessor != null && textProcessor.getMethod() == TextProcessor.VectorizeMethod.EMBEDDING_SEQUENCE) {
                int vocabSize = textProcessor.getVocabSize();
                int embeddingDim = Math.min(128, sizes.length > 0 ? sizes[0] : 128);
                int hiddenDim = sizes.length > 1 ? sizes[1] : 64;
                int outputDim = sizes[sizes.length - 1];
                int seqLength = 50;
                
                enhancedNetwork = new machain.network.EnhancedNetwork(
                    vocabSize, embeddingDim, hiddenDim, outputDim, seqLength
                );
                enhancedNetwork.setLearningRate(Double.parseDouble(lrField.getText().trim()));
                
                outputMode = (OutputMode) outputModeBox.getSelectedItem();
                
                JOptionPane.showMessageDialog(this,
                    "增强网络已构建 (LSTM + Attention):\n" +
                    "词汇量: " + vocabSize + "\n" +
                    "嵌入维度: " + embeddingDim + "\n" +
                    "隐藏维度: " + hiddenDim + "\n" +
                    "输出维度: " + outputDim + "\n" +
                    "序列长度: " + seqLength,
                    "构建成功", JOptionPane.INFORMATION_MESSAGE);
                return;
            }

            Activation act = Activation.valueOf(activationBox.getSelectedItem().toString());
            Activation[] acts = new Activation[sizes.length - 1];
            for (int i = 0; i < acts.length; i++) acts[i] = act;

            network = new Network(sizes, acts);
            network.setLearningRate(Double.parseDouble(lrField.getText().trim()));
            network.setMomentum(Double.parseDouble(momentumField.getText().trim()));
            if (gpuCheckBox != null && gpuCheckBox.isSelected()) {
                network.setUseGPU(true);
            }

            outputMode = (OutputMode) outputModeBox.getSelectedItem();

            config.setLayerSizes(sizes);
            config.setActivation(act.toString());
            config.setLearningRate(network.getLearningRate());
            config.setMomentum(network.getMomentum());

            JOptionPane.showMessageDialog(this,
                    "网络已构建:\n" + network.getArchitecture() +
                            "\n参数量: " + network.countParameters() +
                            "\n输出模式: " + outputMode.getDisplayName(),
                    "构建成功", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "构建失败: " + ex.getMessage(),
                    "错误", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void buildDualNetwork() {
        try {
            int[] structSizes = parseIntArray(structLayerField.getText());
            int[] textSizes = parseIntArray(textLayerField.getText());
            int bridgeDim = Integer.parseInt(bridgeDimField.getText().trim());
            double crossWeight = Double.parseDouble(crossWeightField.getText().trim());

            Activation act = Activation.valueOf(activationBox.getSelectedItem().toString());

            dualNetwork = new DualStreamNetwork(
                    structSizes, act,
                    textSizes, act,
                    bridgeDim
            );
            dualNetwork.setCrossLossWeight(crossWeight);
            dualNetwork.getStructNetwork().setLearningRate(Double.parseDouble(lrField.getText().trim()));
            dualNetwork.getStructNetwork().setMomentum(Double.parseDouble(momentumField.getText().trim()));
            dualNetwork.getTextNetwork().setLearningRate(Double.parseDouble(lrField.getText().trim()));
            dualNetwork.getTextNetwork().setMomentum(Double.parseDouble(momentumField.getText().trim()));
            if (gpuCheckBox != null && gpuCheckBox.isSelected()) {
                dualNetwork.getStructNetwork().setUseGPU(true);
                dualNetwork.getTextNetwork().setUseGPU(true);
            }

            dualMode = true;
            if (modeToggleLabel != null) modeToggleLabel.setText("<html><u>模式: 双流</u></html>");
            JOptionPane.showMessageDialog(this,
                    "双流网络已构建:\n" + dualNetwork.getArchitecture() +
                            "\n总参数量: " + dualNetwork.countParameters() +
                            "\n桥接维度: " + bridgeDim +
                            "\n交叉蒸馏权重: " + crossWeight,
                    "构建成功", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "构建失败: " + ex.getMessage(),
                    "错误", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void saveConfig() {
        try {
            config.setLayerSizes(parseIntArray(layerField.getText()));
            config.setActivation(activationBox.getSelectedItem().toString());
            config.setLearningRate(Double.parseDouble(lrField.getText().trim()));
            config.setMomentum(Double.parseDouble(momentumField.getText().trim()));
            config.saveToFile("machain.properties");
            JOptionPane.showMessageDialog(this, "配置已保存到 machain.properties");
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "保存失败: " + ex.getMessage());
        }
    }

    private void loadConfig() {
        JFileChooser fc = new JFileChooser();
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("Properties", "properties"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            config.loadFromFile(fc.getSelectedFile().getAbsolutePath());
            layerField.setText(arrayToString(config.getLayerSizes()));
            activationBox.setSelectedItem(config.getActivation());
            lrField.setText(String.valueOf(config.getLearningRate()));
            momentumField.setText(String.valueOf(config.getMomentum()));
            JOptionPane.showMessageDialog(this, "配置已加载");
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage());
        }
    }

    private void applyModelParams() {
        try {
            int complexity = Integer.parseInt(complexityField.getText().trim());
            int precision = Integer.parseInt(precisionField.getText().trim());
            double regularization = Double.parseDouble(regularizationField.getText().trim());

            complexitySlider.setValue(Math.min(200, Math.max(1, complexity)));
            precisionSlider.setValue(Math.min(20, Math.max(1, precision)));
            regularizationSlider.setValue((int)(regularization * 1000));

            int baseNeurons = complexity * 8;
            String[] currentLayers = layerField.getText().trim().split(",");
            if (currentLayers.length >= 3) {
                int inputDim = Integer.parseInt(currentLayers[0].trim());
                int outputDim = Integer.parseInt(currentLayers[currentLayers.length - 1].trim());
                int hiddenLayers = currentLayers.length - 2;
                StringBuilder sb = new StringBuilder();
                sb.append(inputDim);
                for (int i = 0; i < hiddenLayers; i++) {
                    int neurons = Math.max(2, baseNeurons - i * (baseNeurons / (hiddenLayers + 1)));
                    sb.append(",").append(neurons);
                }
                sb.append(",").append(outputDim);
                layerField.setText(sb.toString());
            }

            double lr = 0.1 / Math.pow(1.8, precision - 1);
            lr = Math.max(0.00001, Math.min(0.5, lr));
            lrField.setText(String.format("%.6f", lr));

            JOptionPane.showMessageDialog(this,
                    String.format("参数已应用:\n复杂度=%d (基数神经元=%d)\n精密度=%d (lr=%.6f)\n正则化=%.4f",
                            complexity, baseNeurons, precision, lr, regularization),
                    "应用成功", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "参数应用失败: " + ex.getMessage());
        }
    }

    private void resetModelParams() {
        complexitySlider.setValue(10);
        complexityField.setText("10");
        precisionSlider.setValue(5);
        precisionField.setText("5");
        regularizationSlider.setValue(0);
        regularizationField.setText("0.0");
    }

    private void toggleGPU() {
        boolean enable = gpuCheckBox.isSelected();
        try {
            GPUAccelerator acc = GPUAccelerator.getInstance();
            if (enable && !acc.isEnabled()) {
                gpuStatusLabel.setText("GPU不可用: " + acc.getStatus());
                gpuStatusLabel.setForeground(new Color(200, 0, 0));
                gpuCheckBox.setSelected(false);
                return;
            }
            acc.setEnabled(enable);
            if (network != null) {
                network.setUseGPU(enable);
            }
            if (dualNetwork != null) {
                dualNetwork.getStructNetwork().setUseGPU(enable);
                dualNetwork.getTextNetwork().setUseGPU(enable);
            }
            gpuStatusLabel.setText("GPU状态: " + acc.getStatus());
            gpuStatusLabel.setForeground(enable ? new Color(0, 120, 0) : Color.GRAY);
            trainLogArea.append("[GPU] " + (enable ? "已启用CUDA加速\n" : "已切换到CPU模式\n"));
        } catch (Throwable t) {
            gpuCheckBox.setSelected(false);
            gpuStatusLabel.setText("GPU错误: " + t.getMessage());
            gpuStatusLabel.setForeground(new Color(200, 0, 0));
        }
    }

    // ========== 数据管理面板 ==========
    private JPanel buildDataPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel channelPanel = new JPanel();
        channelPanel.setLayout(new BoxLayout(channelPanel, BoxLayout.Y_AXIS));
        channelPanel.setBorder(BorderFactory.createTitledBorder("数据输入通道"));

        JPanel structPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        structPanel.setBorder(BorderFactory.createLineBorder(new Color(0, 100, 0)));
        structPanel.add(new JLabel("通道1 [结构化]:"));
        JButton loadStructBtn = new JButton("加载数值CSV");
        loadStructBtn.addActionListener(e -> loadStructData());
        structPanel.add(loadStructBtn);
        JButton structNormBtn = new JButton("归一化");
        structNormBtn.addActionListener(e -> {
            if (structDataSet != null) { structDataSet.normalize(); structDataLabel.setText("结构化数据: 已归一化"); }
        });
        structPanel.add(structNormBtn);
        JButton structSplitBtn = new JButton("划分");
        structSplitBtn.addActionListener(e -> splitStructData());
        structPanel.add(structSplitBtn);
        channelPanel.add(structPanel);

        JPanel textPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        textPanel.setBorder(BorderFactory.createLineBorder(new Color(0, 0, 150)));
        textPanel.add(new JLabel("通道2 [非结构化]:"));
        JButton loadTextBtn = new JButton("加载文本CSV");
        loadTextBtn.addActionListener(e -> loadTextDataDual());
        textPanel.add(loadTextBtn);
        JButton extractBtn = new JButton("样本提取");
        extractBtn.addActionListener(e -> extractSamples());
        textPanel.add(extractBtn);
        JButton directImportBtn = new JButton("直接导入");
        directImportBtn.addActionListener(e -> directImportTextSamples());
        textPanel.add(directImportBtn);
        JButton textNormBtn = new JButton("向量化");
        textNormBtn.addActionListener(e -> {
            if (textDataSet != null) { textDataSet.normalize(); textDataLabel.setText("文本数据: 已归一化"); }
        });
        textPanel.add(textNormBtn);
        JButton textSplitBtn = new JButton("划分");
        textSplitBtn.addActionListener(e -> splitTextData());
        textPanel.add(textSplitBtn);
        channelPanel.add(textPanel);

        JPanel singlePanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        singlePanel.setBorder(BorderFactory.createTitledBorder("单流模式 (兼容)"));
        JButton loadSingleBtn = new JButton("加载数值CSV");
        loadSingleBtn.addActionListener(e -> loadNumericData());
        singlePanel.add(loadSingleBtn);
        JButton loadSingleTextBtn = new JButton("加载文本CSV");
        loadSingleTextBtn.addActionListener(e -> loadTextData());
        singlePanel.add(loadSingleTextBtn);
        JButton singleExtractBtn = new JButton("样本提取");
        singleExtractBtn.addActionListener(e -> extractSamplesSingle());
        singlePanel.add(singleExtractBtn);
        JButton singleDirectBtn = new JButton("直接导入");
        singleDirectBtn.addActionListener(e -> directImportTextSamplesSingle());
        singlePanel.add(singleDirectBtn);
        JButton normalizeBtn = new JButton("归一化");
        normalizeBtn.addActionListener(e -> normalizeData());
        singlePanel.add(normalizeBtn);
        JButton splitBtn = new JButton("划分训练/测试");
        splitBtn.addActionListener(e -> splitData());
        singlePanel.add(splitBtn);
        dataStatusLabel = new JLabel("未加载数据");
        singlePanel.add(dataStatusLabel);
        channelPanel.add(singlePanel);

        panel.add(new JScrollPane(channelPanel), BorderLayout.NORTH);

        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        dataAnalysisArea = new JTextArea(10, 40);
        dataAnalysisArea.setEditable(false);
        dataAnalysisArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane sp1 = new JScrollPane(dataAnalysisArea);
        sp1.setBorder(BorderFactory.createTitledBorder("数据分析"));

        dataPreviewArea = new JTextArea(8, 40);
        dataPreviewArea.setEditable(false);
        dataPreviewArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane sp2 = new JScrollPane(dataPreviewArea);
        sp2.setBorder(BorderFactory.createTitledBorder("数据预览 (前20行)"));

        splitPane.setTopComponent(sp1);
        splitPane.setBottomComponent(sp2);
        splitPane.setResizeWeight(0.5);
        panel.add(splitPane, BorderLayout.CENTER);

        return panel;
    }

    private void loadStructData() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV文件", "csv"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        String path = fc.getSelectedFile().getAbsolutePath();
        config.setDataPath(fc.getSelectedFile().getParent());

        JPanel inputPanel = new JPanel(new GridLayout(3, 2, 5, 5));
        JTextField inputColsField = new JTextField("2");
        JTextField targetColsField = new JTextField("1");
        JCheckBox headerBox = new JCheckBox("包含表头", true);
        inputPanel.add(new JLabel("输入特征列数:"));
        inputPanel.add(inputColsField);
        inputPanel.add(new JLabel("输出列数:"));
        inputPanel.add(targetColsField);
        inputPanel.add(headerBox);

        if (JOptionPane.showConfirmDialog(this, inputPanel, "结构化数据导入",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int inputCols = Integer.parseInt(inputColsField.getText().trim());
            int targetCols = Integer.parseInt(targetColsField.getText().trim());
            structDataSet = DataSet.loadCSV(path, inputCols, targetCols, headerBox.isSelected());
            structDataLabel.setText("结构化数据: " + structDataSet.size() + " 条 | " +
                    structDataSet.inputDim() + " 输入 | " + structDataSet.outputDim() + " 输出");
            dataAnalysisArea.setText("=== 结构化数据 ===\n" + structDataSet.getAnalysis());
            dataPreviewArea.setText(structDataSet.preview(20));
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage());
        }
    }

    private void splitStructData() {
        if (structDataSet == null) { JOptionPane.showMessageDialog(this, "请先加载结构化数据"); return; }
        DataSet[] splits = structDataSet.split(0.8);
        structTrainSet = splits[0]; structTestSet = splits[1];
        JOptionPane.showMessageDialog(this, "结构化: 训练集 " + structTrainSet.size() + ", 测试集 " + structTestSet.size());
    }

    private void loadTextDataDual() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV文件", "csv"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        String path = fc.getSelectedFile().getAbsolutePath();
        config.setDataPath(fc.getSelectedFile().getParent());

        JPanel inputPanel = new JPanel(new GridLayout(6, 2, 5, 5));
        JTextField textColField = new JTextField("0");
        JTextField labelColField = new JTextField("1");
        JCheckBox headerBox = new JCheckBox("包含表头", true);
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JComboBox<String> chunkingBox = new JComboBox<>(SampleExtractor.chunkingStrategyLabels());
        JSlider noiseFilterSlider = new JSlider(0, 10, 0);
        noiseFilterSlider.setMajorTickSpacing(2);
        noiseFilterSlider.setMinorTickSpacing(1);
        noiseFilterSlider.setPaintTicks(true);
        noiseFilterSlider.setPaintLabels(true);
        JLabel noiseFilterValueLabel = new JLabel("0");
        noiseFilterSlider.addChangeListener(e -> noiseFilterValueLabel.setText(String.valueOf(noiseFilterSlider.getValue())));
        
        inputPanel.add(new JLabel("文本列索引:"));
        inputPanel.add(textColField);
        inputPanel.add(new JLabel("标签列索引:"));
        inputPanel.add(labelColField);
        inputPanel.add(new JLabel("向量化方法:"));
        inputPanel.add(methodBox);
        inputPanel.add(new JLabel("分块策略:"));
        inputPanel.add(chunkingBox);
        inputPanel.add(new JLabel("噪声过滤 (0-10):"));
        JPanel noisePanel = new JPanel(new BorderLayout(5, 0));
        noisePanel.add(noiseFilterSlider, BorderLayout.CENTER);
        noisePanel.add(noiseFilterValueLabel, BorderLayout.EAST);
        inputPanel.add(noisePanel);
        inputPanel.add(headerBox);

        if (JOptionPane.showConfirmDialog(this, inputPanel, "文本数据导入",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int textCol = Integer.parseInt(textColField.getText().trim());
            int labelCol = Integer.parseInt(labelColField.getText().trim());
            int noiseLevel = noiseFilterSlider.getValue();
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            structTextProcessor = new TextProcessor(method, 1000);
            textDataSet = DataSet.loadTextCSV(path, textCol, labelCol, headerBox.isSelected(), structTextProcessor);
            
            if (noiseLevel > 0) {
                machain.data.NoiseFilter filter = new machain.data.NoiseFilter(noiseLevel);
                double[][] filteredInputs = filter.filterInputs(textDataSet.getInputs(), noiseLevel);
                double[][] filteredTargets = new double[filteredInputs.length][textDataSet.getTargets()[0].length];
                for (int i = 0; i < filteredInputs.length; i++) {
                    filteredTargets[i] = textDataSet.getTargets()[i];
                }
                textDataSet = DataSet.fromArrays(filteredInputs, filteredTargets, 
                    textDataSet.getFeatureNames(), textDataSet.getLabelNames());
            }
            
            textDataLabel.setText("文本数据: " + textDataSet.size() + " 条 | 词汇量: " +
                    structTextProcessor.getVectorSize() + " | " + method + " | 噪声过滤: " + noiseLevel);

            textLayerField.setText(structTextProcessor.getVectorSize() + ",32," + textDataSet.outputDim());

            dataAnalysisArea.setText("=== 文本数据 ===\n" + textDataSet.getAnalysis());
            dataPreviewArea.setText(textDataSet.preview(20));
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage());
        }
    }

    private void extractSamples() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                "文本文件", "txt", "csv", "tsv", "md", "text", "json"));
        fc.setMultiSelectionEnabled(true);
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File[] files = fc.getSelectedFiles();
        config.setDataPath(files[0].getParent());

        JPanel configPanel = new JPanel(new GridLayout(0, 2, 5, 5));
        JComboBox<String> strategyBox = new JComboBox<>(SampleExtractor.strategyLabels());
        JComboBox<String> chunkingBox = new JComboBox<>(SampleExtractor.chunkingStrategyLabels());
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JTextField vocabField = new JTextField("1000");
        JTextField paramField = new JTextField("");
        JSlider noiseFilterSlider = new JSlider(0, 10, 0);
        noiseFilterSlider.setMajorTickSpacing(2);
        noiseFilterSlider.setMinorTickSpacing(1);
        noiseFilterSlider.setPaintTicks(true);
        noiseFilterSlider.setPaintLabels(true);
        JLabel noiseFilterValueLabel = new JLabel("0");
        noiseFilterSlider.addChangeListener(e -> noiseFilterValueLabel.setText(String.valueOf(noiseFilterSlider.getValue())));

        configPanel.add(new JLabel("提取策略:"));
        configPanel.add(strategyBox);
        configPanel.add(new JLabel("分块策略:"));
        configPanel.add(chunkingBox);
        configPanel.add(new JLabel("向量化方法:"));
        configPanel.add(methodBox);
        configPanel.add(new JLabel("最大词汇量:"));
        configPanel.add(vocabField);
        configPanel.add(new JLabel("附加参数:"));
        configPanel.add(paramField);
        configPanel.add(new JLabel("噪声过滤 (0-10):"));
        JPanel noisePanel = new JPanel(new BorderLayout(5, 0));
        noisePanel.add(noiseFilterSlider, BorderLayout.CENTER);
        noisePanel.add(noiseFilterValueLabel, BorderLayout.EAST);
        configPanel.add(noisePanel);

        JTextArea descArea = new JTextArea(3, 30);
        descArea.setEditable(false);
        descArea.setLineWrap(true);
        descArea.setWrapStyleWord(true);
        descArea.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        descArea.setText(SampleExtractor.Strategy.ARTICLE_TOPIC.description());
        JScrollPane descScroll = new JScrollPane(descArea);
        descScroll.setBorder(BorderFactory.createTitledBorder("策略说明"));

        strategyBox.addActionListener(e -> {
            SampleExtractor.Strategy s = SampleExtractor.fromLabel(strategyBox.getSelectedItem().toString());
            descArea.setText(s.description());
            switch (s) {
                case BILINGUAL_PAIR: paramField.setText("direction=en2zh"); break;
                case SEQUENCE_PREDICT: paramField.setText("contextSize=2"); break;
                case KEYWORD_PREDICT: paramField.setText("topK=3"); break;
                case AUGMENT: paramField.setText("factor=3"); break;
                default: paramField.setText(""); break;
            }
        });

        JPanel allConfig = new JPanel(new BorderLayout(5, 5));
        allConfig.add(configPanel, BorderLayout.NORTH);
        allConfig.add(descScroll, BorderLayout.CENTER);

        if (JOptionPane.showConfirmDialog(this, allConfig, "样本提取配置 (" + files.length + " 个文件)",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            SampleExtractor.Strategy strategy = SampleExtractor.fromLabel(strategyBox.getSelectedItem().toString());
            SampleExtractor extractor = new SampleExtractor(strategy);
            
            SampleExtractor.ChunkingStrategy chunkingStrategy = SampleExtractor.ChunkingStrategy.fromLabel(
                chunkingBox.getSelectedItem().toString());
            extractor.setChunkingStrategy(chunkingStrategy);

            String paramStr = paramField.getText().trim();
            if (!paramStr.isEmpty()) {
                for (String pair : paramStr.split("\\s*;\\s*")) {
                    String[] kv = pair.split("\\s*=\\s*", 2);
                    if (kv.length == 2) extractor.setParam(kv[0].trim(), kv[1].trim());
                }
            }

            List<SampleExtractor.ExtractedSample> allSamples = new ArrayList<>();
            StringBuilder loadLog = new StringBuilder();

            for (File file : files) {
                List<String> lines = java.nio.file.Files.readAllLines(file.toPath(),
                        java.nio.charset.StandardCharsets.UTF_8);

                List<SampleExtractor.ExtractedSample> samples;
                if (strategy == SampleExtractor.Strategy.BILINGUAL_PAIR
                        || strategy == SampleExtractor.Strategy.ARTICLE_TOPIC) {
                    samples = extractor.extractFromLines(lines);
                } else {
                    String fullText = String.join("\n", lines);
                    samples = extractor.extract(Collections.singletonList(fullText));
                }

                allSamples.addAll(samples);
                loadLog.append(file.getName()).append(": ").append(samples.size()).append(" 条样本\n");
            }

            if (allSamples.isEmpty()) {
                JOptionPane.showMessageDialog(this, "未提取到任何样本，请检查文件格式和策略选择");
                return;
            }

            int noiseLevel = noiseFilterSlider.getValue();
            if (noiseLevel > 0) {
                int originalCount = allSamples.size();
                machain.data.NoiseFilter filter = new machain.data.NoiseFilter(noiseLevel);
                allSamples = filter.filter(allSamples);
                loadLog.append("[噪声过滤] ").append(originalCount).append(" -> ")
                    .append(allSamples.size()).append(" 条样本 (过滤等级: ").append(noiseLevel).append(")\n");
            }

            originalTexts.clear();
            for (SampleExtractor.ExtractedSample s : allSamples) {
                if (s.inputText != null && !s.inputText.trim().isEmpty()) {
                    originalTexts.add(s.inputText.trim());
                }
            }

            int maxVocab = Integer.parseInt(vocabField.getText().trim());
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            structTextProcessor = new TextProcessor(method, maxVocab);
            textDataSet = extractor.toDataSet(allSamples, structTextProcessor);

            int outputDim = textDataSet.outputDim();
            textLayerField.setText(structTextProcessor.getVectorSize() + ",32," + outputDim);

            String statusMsg = "样本提取: " + allSamples.size() + " 条 | 策略: " + strategy.label
                    + " | 词汇量: " + structTextProcessor.getVectorSize() + " | " + method;
            textDataLabel.setText(statusMsg);

            dataAnalysisArea.setText("=== 样本提取结果 ===\n" + loadLog.toString()
                    + "\n总计: " + allSamples.size() + " 条样本\n"
                    + "向量化维度: " + structTextProcessor.getVectorSize()
                    + "\n输出维度: " + outputDim
                    + "\n\n" + textDataSet.getAnalysis());
            dataPreviewArea.setText(textDataSet.preview(20));

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "样本提取失败: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void directImportTextSamples() {
        String[] modes = {"从文件导入 (TSV/CSV)", "手动粘贴输入"};
        int choice = JOptionPane.showOptionDialog(this, "选择导入方式", "直接导入文本样本",
                JOptionPane.DEFAULT_OPTION, JOptionPane.PLAIN_MESSAGE, null, modes, modes[0]);
        if (choice < 0) return;

        List<SampleExtractor.ExtractedSample> samples = new ArrayList<>();

        if (choice == 0) {
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                    "文本文件", "txt", "tsv", "csv", "md"));
            fc.setMultiSelectionEnabled(true);
            if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
            File[] files = fc.getSelectedFiles();
            config.setDataPath(files[0].getParent());

            JPanel cfgPanel = new JPanel(new GridLayout(3, 2, 5, 5));
            JTextField sepField = new JTextField("\\t");
            JTextField inputIdxField = new JTextField("0");
            JTextField labelIdxField = new JTextField("1");
            cfgPanel.add(new JLabel("分隔符 (正则):"));
            cfgPanel.add(sepField);
            cfgPanel.add(new JLabel("输入文本列索引:"));
            cfgPanel.add(inputIdxField);
            cfgPanel.add(new JLabel("标签列索引:"));
            cfgPanel.add(labelIdxField);

            if (JOptionPane.showConfirmDialog(this, cfgPanel, "文件解析设置",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

            String sep = sepField.getText().replace("\\t", "\t");
            int inputIdx = Integer.parseInt(inputIdxField.getText().trim());
            int labelIdx = Integer.parseInt(labelIdxField.getText().trim());

            try {
                for (File file : files) {
                    List<String> lines = java.nio.file.Files.readAllLines(file.toPath(),
                            java.nio.charset.StandardCharsets.UTF_8);
                    int count = 0;
                    for (String line : lines) {
                        if (line.trim().isEmpty()) continue;
                        String[] cols = line.split(sep, -1);
                        if (Math.max(inputIdx, labelIdx) >= cols.length) continue;
                        samples.add(new SampleExtractor.ExtractedSample(
                                cols[inputIdx].trim(), cols[labelIdx].trim()));
                        count++;
                    }
                    trainLogArea.append("[直接导入] " + file.getName() + ": " + count + " 条\n");
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "文件导入失败: " + ex.getMessage());
                return;
            }
        } else {
            JTextArea inputArea = new JTextArea(15, 50);
            inputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            inputArea.setLineWrap(true);
            JScrollPane sp = new JScrollPane(inputArea);
            sp.setBorder(BorderFactory.createTitledBorder(
                    "每行一条样本，格式: 输入文本<TAB>标签\n示例:\n这部电影很好看\t正面\n今天天气不错\t正面\n电脑死机了\t负面"));

            if (JOptionPane.showConfirmDialog(this, sp, "粘贴文本样本",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

            String text = inputArea.getText();
            if (text.trim().isEmpty()) return;
            for (String line : text.split("\n")) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split("\t", 2);
                if (parts.length < 2) {
                    samples.add(new SampleExtractor.ExtractedSample(parts[0].trim(), "未知"));
                } else {
                    samples.add(new SampleExtractor.ExtractedSample(parts[0].trim(), parts[1].trim()));
                }
            }
        }

        if (samples.isEmpty()) {
            JOptionPane.showMessageDialog(this, "未导入任何样本");
            return;
        }

        originalTexts.clear();
        for (SampleExtractor.ExtractedSample s : samples) {
            if (s.inputText != null && !s.inputText.trim().isEmpty()) {
                originalTexts.add(s.inputText.trim());
            }
        }

        JPanel vecPanel = new JPanel(new GridLayout(2, 2, 5, 5));
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JTextField vocabField = new JTextField("1000");
        vecPanel.add(new JLabel("向量化方法:"));
        vecPanel.add(methodBox);
        vecPanel.add(new JLabel("最大词汇量:"));
        vecPanel.add(vocabField);

        if (JOptionPane.showConfirmDialog(this, vecPanel, "向量化设置 (" + samples.size() + " 条样本)",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int maxVocab = Integer.parseInt(vocabField.getText().trim());
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            structTextProcessor = new TextProcessor(method, maxVocab);
            SampleExtractor helper = new SampleExtractor(SampleExtractor.Strategy.ARTICLE_TOPIC);
            textDataSet = helper.toDataSet(samples, structTextProcessor);

            int outputDim = textDataSet.outputDim();
            textLayerField.setText(structTextProcessor.getVectorSize() + ",32," + outputDim);

            textDataLabel.setText("直接导入: " + samples.size() + " 条 | " + method
                    + " | 词汇量: " + structTextProcessor.getVectorSize());
            dataAnalysisArea.setText("=== 直接导入结果 ===\n总计: " + samples.size() + " 条样本\n"
                    + "向量化维度: " + structTextProcessor.getVectorSize()
                    + "\n输出维度: " + outputDim
                    + "\n\n" + textDataSet.getAnalysis());
            dataPreviewArea.setText(textDataSet.preview(20));
            trainLogArea.append("[直接导入] 完成: " + samples.size() + " 条样本, "
                    + method + ", 词汇量=" + structTextProcessor.getVectorSize() + "\n");
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "导入处理失败: " + ex.getMessage());
        }
    }

    private void extractSamplesSingle() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                "文本文件", "txt", "csv", "tsv", "md", "text", "json"));
        fc.setMultiSelectionEnabled(true);
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File[] files = fc.getSelectedFiles();
        config.setDataPath(files[0].getParent());

        JPanel configPanel = new JPanel(new GridLayout(0, 2, 5, 5));
        JComboBox<String> strategyBox = new JComboBox<>(SampleExtractor.strategyLabels());
        JComboBox<String> chunkingBox = new JComboBox<>(SampleExtractor.chunkingStrategyLabels());
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JTextField vocabField = new JTextField("1000");
        JTextField paramField = new JTextField("");
        JSlider noiseFilterSlider = new JSlider(0, 10, 0);
        noiseFilterSlider.setMajorTickSpacing(2);
        noiseFilterSlider.setMinorTickSpacing(1);
        noiseFilterSlider.setPaintTicks(true);
        noiseFilterSlider.setPaintLabels(true);
        JLabel noiseFilterValueLabel = new JLabel("0");
        noiseFilterSlider.addChangeListener(e -> noiseFilterValueLabel.setText(String.valueOf(noiseFilterSlider.getValue())));

        configPanel.add(new JLabel("提取策略:"));
        configPanel.add(strategyBox);
        configPanel.add(new JLabel("分块策略:"));
        configPanel.add(chunkingBox);
        configPanel.add(new JLabel("向量化方法:"));
        configPanel.add(methodBox);
        configPanel.add(new JLabel("最大词汇量:"));
        configPanel.add(vocabField);
        configPanel.add(new JLabel("附加参数:"));
        configPanel.add(paramField);
        configPanel.add(new JLabel("噪声过滤 (0-10):"));
        JPanel noisePanel = new JPanel(new BorderLayout(5, 0));
        noisePanel.add(noiseFilterSlider, BorderLayout.CENTER);
        noisePanel.add(noiseFilterValueLabel, BorderLayout.EAST);
        configPanel.add(noisePanel);

        JTextArea descArea = new JTextArea(3, 30);
        descArea.setEditable(false);
        descArea.setLineWrap(true);
        descArea.setWrapStyleWord(true);
        descArea.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        descArea.setText(SampleExtractor.Strategy.ARTICLE_TOPIC.description());
        JScrollPane descScroll = new JScrollPane(descArea);
        descScroll.setBorder(BorderFactory.createTitledBorder("策略说明"));

        strategyBox.addActionListener(e -> {
            SampleExtractor.Strategy s = SampleExtractor.fromLabel(strategyBox.getSelectedItem().toString());
            descArea.setText(s.description());
            switch (s) {
                case BILINGUAL_PAIR: paramField.setText("direction=en2zh"); break;
                case SEQUENCE_PREDICT: paramField.setText("contextSize=2"); break;
                case KEYWORD_PREDICT: paramField.setText("topK=3"); break;
                case AUGMENT: paramField.setText("factor=3"); break;
                default: paramField.setText(""); break;
            }
        });

        JPanel allConfig = new JPanel(new BorderLayout(5, 5));
        allConfig.add(configPanel, BorderLayout.NORTH);
        allConfig.add(descScroll, BorderLayout.CENTER);

        if (JOptionPane.showConfirmDialog(this, allConfig, "样本提取配置 (单流模式, " + files.length + " 个文件)",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            SampleExtractor.Strategy strategy = SampleExtractor.fromLabel(strategyBox.getSelectedItem().toString());
            SampleExtractor extractor = new SampleExtractor(strategy);
            
            SampleExtractor.ChunkingStrategy chunkingStrategy = SampleExtractor.ChunkingStrategy.fromLabel(
                chunkingBox.getSelectedItem().toString());
            extractor.setChunkingStrategy(chunkingStrategy);

            String paramStr = paramField.getText().trim();
            if (!paramStr.isEmpty()) {
                for (String pair : paramStr.split("\\s*;\\s*")) {
                    String[] kv = pair.split("\\s*=\\s*", 2);
                    if (kv.length == 2) extractor.setParam(kv[0].trim(), kv[1].trim());
                }
            }

            List<SampleExtractor.ExtractedSample> allSamples = new ArrayList<>();
            StringBuilder loadLog = new StringBuilder();

            for (File file : files) {
                List<String> lines = java.nio.file.Files.readAllLines(file.toPath(),
                        java.nio.charset.StandardCharsets.UTF_8);
                List<SampleExtractor.ExtractedSample> samples;
                if (strategy == SampleExtractor.Strategy.BILINGUAL_PAIR
                        || strategy == SampleExtractor.Strategy.ARTICLE_TOPIC) {
                    samples = extractor.extractFromLines(lines);
                } else {
                    String fullText = String.join("\n", lines);
                    samples = extractor.extract(Collections.singletonList(fullText));
                }
                allSamples.addAll(samples);
                loadLog.append(file.getName()).append(": ").append(samples.size()).append(" 条样本\n");
            }

            if (allSamples.isEmpty()) {
                JOptionPane.showMessageDialog(this, "未提取到任何样本");
                return;
            }

            int noiseLevel = noiseFilterSlider.getValue();
            if (noiseLevel > 0) {
                int originalCount = allSamples.size();
                machain.data.NoiseFilter filter = new machain.data.NoiseFilter(noiseLevel);
                allSamples = filter.filter(allSamples);
                loadLog.append("[噪声过滤] ").append(originalCount).append(" -> ")
                    .append(allSamples.size()).append(" 条样本 (过滤等级: ").append(noiseLevel).append(")\n");
            }

            originalTexts.clear();
            for (SampleExtractor.ExtractedSample s : allSamples) {
                if (s.inputText != null && !s.inputText.trim().isEmpty()) {
                    originalTexts.add(s.inputText.trim());
                }
            }

            int maxVocab = Integer.parseInt(vocabField.getText().trim());
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            textProcessor = new TextProcessor(method, maxVocab);
            dataSet = extractor.toDataSet(allSamples, textProcessor);

            int outputDim = dataSet.outputDim();
            layerField.setText(textProcessor.getVectorSize() + ",32," + outputDim);

            String statusMsg = "单流样本提取: " + allSamples.size() + " 条 | " + strategy.label
                    + " | " + method + " | 词汇量: " + textProcessor.getVectorSize();
            dataStatusLabel.setText(statusMsg);

            dataAnalysisArea.setText("=== 单流样本提取结果 ===\n" + loadLog.toString()
                    + "\n总计: " + allSamples.size() + " 条样本\n"
                    + "向量化维度: " + textProcessor.getVectorSize()
                    + "\n输出维度: " + outputDim
                    + "\n\n" + dataSet.getAnalysis());
            dataPreviewArea.setText(dataSet.preview(20));
            trainLogArea.append("[单流样本提取] " + allSamples.size() + " 条, 策略=" + strategy.label + "\n");

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "样本提取失败: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void directImportTextSamplesSingle() {
        String[] modes = {"从文件导入 (TSV/CSV)", "手动粘贴输入"};
        int choice = JOptionPane.showOptionDialog(this, "选择导入方式", "直接导入文本样本 (单流模式)",
                JOptionPane.DEFAULT_OPTION, JOptionPane.PLAIN_MESSAGE, null, modes, modes[0]);
        if (choice < 0) return;

        List<SampleExtractor.ExtractedSample> samples = new ArrayList<>();

        if (choice == 0) {
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                    "文本文件", "txt", "tsv", "csv", "md"));
            fc.setMultiSelectionEnabled(true);
            if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
            File[] files = fc.getSelectedFiles();
            config.setDataPath(files[0].getParent());

            JPanel cfgPanel = new JPanel(new GridLayout(3, 2, 5, 5));
            JTextField sepField = new JTextField("\\t");
            JTextField inputIdxField = new JTextField("0");
            JTextField labelIdxField = new JTextField("1");
            cfgPanel.add(new JLabel("分隔符 (正则):"));
            cfgPanel.add(sepField);
            cfgPanel.add(new JLabel("输入文本列索引:"));
            cfgPanel.add(inputIdxField);
            cfgPanel.add(new JLabel("标签列索引:"));
            cfgPanel.add(labelIdxField);

            if (JOptionPane.showConfirmDialog(this, cfgPanel, "文件解析设置",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

            String sep = sepField.getText().replace("\\t", "\t");
            int inputIdx = Integer.parseInt(inputIdxField.getText().trim());
            int labelIdx = Integer.parseInt(labelIdxField.getText().trim());

            try {
                for (File file : files) {
                    List<String> lines = java.nio.file.Files.readAllLines(file.toPath(),
                            java.nio.charset.StandardCharsets.UTF_8);
                    int count = 0;
                    for (String line : lines) {
                        if (line.trim().isEmpty()) continue;
                        String[] cols = line.split(sep, -1);
                        if (Math.max(inputIdx, labelIdx) >= cols.length) continue;
                        samples.add(new SampleExtractor.ExtractedSample(
                                cols[inputIdx].trim(), cols[labelIdx].trim()));
                        count++;
                    }
                    trainLogArea.append("[单流直接导入] " + file.getName() + ": " + count + " 条\n");
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "文件导入失败: " + ex.getMessage());
                return;
            }
        } else {
            JTextArea inputArea = new JTextArea(15, 50);
            inputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            inputArea.setLineWrap(true);
            JScrollPane sp = new JScrollPane(inputArea);
            sp.setBorder(BorderFactory.createTitledBorder(
                    "每行一条样本，格式: 输入文本<TAB>标签"));

            if (JOptionPane.showConfirmDialog(this, sp, "粘贴文本样本 (单流模式)",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

            String text = inputArea.getText();
            if (text.trim().isEmpty()) return;
            for (String line : text.split("\n")) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split("\t", 2);
                if (parts.length < 2) {
                    samples.add(new SampleExtractor.ExtractedSample(parts[0].trim(), "未知"));
                } else {
                    samples.add(new SampleExtractor.ExtractedSample(parts[0].trim(), parts[1].trim()));
                }
            }
        }

        if (samples.isEmpty()) {
            JOptionPane.showMessageDialog(this, "未导入任何样本");
            return;
        }

        originalTexts.clear();
        for (SampleExtractor.ExtractedSample s : samples) {
            if (s.inputText != null && !s.inputText.trim().isEmpty()) {
                originalTexts.add(s.inputText.trim());
            }
        }

        JPanel vecPanel = new JPanel(new GridLayout(2, 2, 5, 5));
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JTextField vocabField = new JTextField("1000");
        vecPanel.add(new JLabel("向量化方法:"));
        vecPanel.add(methodBox);
        vecPanel.add(new JLabel("最大词汇量:"));
        vecPanel.add(vocabField);

        if (JOptionPane.showConfirmDialog(this, vecPanel, "向量化设置 (" + samples.size() + " 条样本)",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int maxVocab = Integer.parseInt(vocabField.getText().trim());
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            textProcessor = new TextProcessor(method, maxVocab);
            SampleExtractor helper = new SampleExtractor(SampleExtractor.Strategy.ARTICLE_TOPIC);
            dataSet = helper.toDataSet(samples, textProcessor);

            int outputDim = dataSet.outputDim();
            layerField.setText(textProcessor.getVectorSize() + ",32," + outputDim);

            dataStatusLabel.setText("单流直接导入: " + samples.size() + " 条 | " + method
                    + " | 词汇量: " + textProcessor.getVectorSize());
            dataAnalysisArea.setText("=== 单流直接导入结果 ===\n总计: " + samples.size() + " 条样本\n"
                    + "向量化维度: " + textProcessor.getVectorSize()
                    + "\n输出维度: " + outputDim
                    + "\n\n" + dataSet.getAnalysis());
            dataPreviewArea.setText(dataSet.preview(20));
            trainLogArea.append("[单流直接导入] 完成: " + samples.size() + " 条样本\n");
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "导入处理失败: " + ex.getMessage());
        }
    }

    private void splitTextData() {
        if (textDataSet == null) { JOptionPane.showMessageDialog(this, "请先加载文本数据"); return; }
        DataSet[] splits = textDataSet.split(0.8);
        textTrainSet = splits[0]; textTestSet = splits[1];
        JOptionPane.showMessageDialog(this, "文本: 训练集 " + textTrainSet.size() + ", 测试集 " + textTestSet.size());
    }

    private void loadNumericData() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV文件", "csv"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        String path = fc.getSelectedFile().getAbsolutePath();
        config.setDataPath(fc.getSelectedFile().getParent());

        JPanel inputPanel = new JPanel(new GridLayout(4, 2, 5, 5));
        JTextField inputColsField = new JTextField("2");
        JTextField targetColsField = new JTextField("1");
        JCheckBox headerBox = new JCheckBox("包含表头", true);
        JCheckBox classBox = new JCheckBox("分类模式(标签为类别名)", false);
        inputPanel.add(new JLabel("输入特征列数:"));
        inputPanel.add(inputColsField);
        inputPanel.add(new JLabel("输出列数(回归模式):"));
        inputPanel.add(targetColsField);
        inputPanel.add(headerBox);
        inputPanel.add(classBox);

        if (JOptionPane.showConfirmDialog(this, inputPanel, "CSV导入设置",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int inputCols = Integer.parseInt(inputColsField.getText().trim());
            boolean hasHeader = headerBox.isSelected();

            if (classBox.isSelected()) {
                dataSet = DataSet.loadClassificationCSV(path, inputCols, hasHeader);
            } else {
                int targetCols = Integer.parseInt(targetColsField.getText().trim());
                dataSet = DataSet.loadCSV(path, inputCols, targetCols, hasHeader);
            }

            textProcessor = null;
            updateDataDisplay();
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage(),
                    "错误", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void loadTextData() {
        JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV文件", "csv"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        String path = fc.getSelectedFile().getAbsolutePath();
        config.setDataPath(fc.getSelectedFile().getParent());

        JPanel inputPanel = new JPanel(new GridLayout(6, 2, 5, 5));
        JTextField textColField = new JTextField("0");
        JTextField labelColField = new JTextField("1");
        JCheckBox headerBox = new JCheckBox("包含表头", true);
        JComboBox<String> methodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM", "EMBEDDING_SEQUENCE"});
        JComboBox<String> chunkingBox = new JComboBox<>(SampleExtractor.chunkingStrategyLabels());
        JSlider noiseFilterSlider = new JSlider(0, 10, 0);
        noiseFilterSlider.setMajorTickSpacing(2);
        noiseFilterSlider.setMinorTickSpacing(1);
        noiseFilterSlider.setPaintTicks(true);
        noiseFilterSlider.setPaintLabels(true);
        JLabel noiseFilterValueLabel = new JLabel("0");
        noiseFilterSlider.addChangeListener(e -> noiseFilterValueLabel.setText(String.valueOf(noiseFilterSlider.getValue())));
        
        inputPanel.add(new JLabel("文本列索引:"));
        inputPanel.add(textColField);
        inputPanel.add(new JLabel("标签列索引:"));
        inputPanel.add(labelColField);
        inputPanel.add(new JLabel("向量化方法:"));
        inputPanel.add(methodBox);
        inputPanel.add(new JLabel("分块策略:"));
        inputPanel.add(chunkingBox);
        inputPanel.add(new JLabel("噪声过滤 (0-10):"));
        JPanel noisePanel = new JPanel(new BorderLayout(5, 0));
        noisePanel.add(noiseFilterSlider, BorderLayout.CENTER);
        noisePanel.add(noiseFilterValueLabel, BorderLayout.EAST);
        inputPanel.add(noisePanel);
        inputPanel.add(headerBox);

        if (JOptionPane.showConfirmDialog(this, inputPanel, "文本CSV导入设置",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) return;

        try {
            int textCol = Integer.parseInt(textColField.getText().trim());
            int labelCol = Integer.parseInt(labelColField.getText().trim());
            boolean hasHeader = headerBox.isSelected();
            int noiseLevel = noiseFilterSlider.getValue();
            TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                    methodBox.getSelectedItem().toString());

            textProcessor = new TextProcessor(method, 1000);
            dataSet = DataSet.loadTextCSV(path, textCol, labelCol, hasHeader, textProcessor);
            
            if (noiseLevel > 0) {
                machain.data.NoiseFilter filter = new machain.data.NoiseFilter(noiseLevel);
                double[][] filteredInputs = filter.filterInputs(dataSet.getInputs(), noiseLevel);
                double[][] filteredTargets = new double[filteredInputs.length][dataSet.getTargets()[0].length];
                for (int i = 0; i < filteredInputs.length; i++) {
                    filteredTargets[i] = dataSet.getTargets()[i];
                }
                dataSet = DataSet.fromArrays(filteredInputs, filteredTargets, 
                    dataSet.getFeatureNames(), dataSet.getLabelNames());
            }

            updateDataDisplay();
            JOptionPane.showMessageDialog(this,
                    "文本数据已加载:\n" +
                            "词汇表大小: " + textProcessor.getVectorSize() + "\n" +
                            "向量化方法: " + method + "\n" +
                            "噪声过滤等级: " + noiseLevel);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage(),
                    "错误", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void updateDataDisplay() {
        dataAnalysisArea.setText(dataSet.getAnalysis());
        dataPreviewArea.setText(dataSet.preview(20));
        String type = textProcessor != null ? "文本" : "数值";
        dataStatusLabel.setText("已加载: " + dataSet.size() + " 条" + type + "样本 | " +
                dataSet.inputDim() + " 输入 | " + dataSet.outputDim() + " 输出");
        trainSet = null;
        testSet = null;
    }

    private void normalizeData() {
        if (dataSet == null) {
            JOptionPane.showMessageDialog(this, "请先加载数据");
            return;
        }
        dataSet.normalize();
        updateDataDisplay();
    }

    private void splitData() {
        if (dataSet == null) {
            JOptionPane.showMessageDialog(this, "请先加载数据");
            return;
        }
        String ratioStr = JOptionPane.showInputDialog(this, "训练集比例:", String.valueOf(config.getTrainSplitRatio()));
        if (ratioStr == null) return;
        try {
            double ratio = Double.parseDouble(ratioStr);
            config.setTrainSplitRatio(ratio);
            DataSet[] splits = dataSet.split(ratio);
            trainSet = splits[0];
            testSet = splits[1];
            JOptionPane.showMessageDialog(this,
                    String.format("已划分: 训练集 %d 条, 测试集 %d 条", trainSet.size(), testSet.size()));
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "比例格式错误");
        }
    }

    // ========== 训练面板 ==========
    private JPanel buildTrainPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        topPanel.add(new JLabel("训练轮数:"));
        epochField = new JTextField("100", 6);
        topPanel.add(epochField);
        topPanel.add(new JLabel("日志间隔:"));
        logIntervalField = new JTextField(String.valueOf(config.getLogInterval()), 4);
        topPanel.add(logIntervalField);

        startBtn = new JButton("开始训练");
        startBtn.addActionListener(e -> startTraining());
        topPanel.add(startBtn);

        stopBtn = new JButton("停止");
        stopBtn.setEnabled(false);
        stopBtn.addActionListener(e -> stopTraining());
        topPanel.add(stopBtn);

        progressBar = new JProgressBar(0, 100);
        progressBar.setPreferredSize(new Dimension(200, 20));
        progressBar.setStringPainted(true);
        topPanel.add(progressBar);

        JPanel regPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        regPanel.setBorder(BorderFactory.createTitledBorder("正则化与优化"));
        regPanel.add(new JLabel("权重衰减:"));
        weightDecayField = new JTextField("0.0001", 6);
        regPanel.add(weightDecayField);
        regPanel.add(new JLabel("Dropout:"));
        dropoutField = new JTextField("0.2", 4);
        regPanel.add(dropoutField);
        regPanel.add(new JLabel("Batch:"));
        batchSizeField = new JTextField("32", 4);
        regPanel.add(batchSizeField);
        regPanel.add(new JLabel("LR调度:"));
        lrScheduleBox = new JComboBox<>(new String[]{"none", "cosine", "step"});
        lrScheduleBox.setSelectedItem("cosine");
        regPanel.add(lrScheduleBox);
        regPanel.add(new JLabel("早停:"));
        earlyStopField = new JTextField("15", 4);
        regPanel.add(earlyStopField);
        regPanel.add(new JLabel("轮"));

        JPanel specPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        specPanel.setBorder(BorderFactory.createTitledBorder("学习特化增强程度"));
        specPanel.add(new JLabel("0=纯背诵"));
        learnSpecSlider = new JSlider(0, 100, 10);
        learnSpecSlider.setPreferredSize(new Dimension(200, 20));
        learnSpecSlider.setMajorTickSpacing(20);
        learnSpecSlider.setMinorTickSpacing(5);
        learnSpecSlider.setPaintTicks(true);
        learnSpecValueLabel = new JLabel("1.0");
        learnSpecValueLabel.setFont(learnSpecValueLabel.getFont().deriveFont(Font.BOLD, 12f));
        learnSpecValueLabel.setForeground(new Color(0, 100, 180));
        learnSpecSlider.addChangeListener(e -> {
            learnSpecCoeff = learnSpecSlider.getValue() / 10.0;
            learnSpecValueLabel.setText(String.format("%.1f", learnSpecCoeff));
            if (learnSpecCoeff == 0) {
                learnSpecValueLabel.setForeground(Color.GRAY);
                learnSpecValueLabel.setText("纯背诵");
            } else if (learnSpecCoeff >= 5) {
                learnSpecValueLabel.setForeground(new Color(200, 0, 0));
            } else {
                learnSpecValueLabel.setForeground(new Color(0, 100, 180));
            }
        });
        specPanel.add(learnSpecSlider);
        specPanel.add(learnSpecValueLabel);
        specPanel.add(new JLabel("10=最大泛化"));
        JLabel specHint = new JLabel("<html><font color='gray' size='2'>影响训练时泛化vs背诵的侧重</font></html>");
        specPanel.add(specHint);

        JPanel northContainer = new JPanel(new BorderLayout());
        northContainer.add(topPanel, BorderLayout.NORTH);
        JPanel regContainer = new JPanel(new BorderLayout());
        regContainer.add(regPanel, BorderLayout.NORTH);
        regContainer.add(specPanel, BorderLayout.CENTER);
        northContainer.add(regContainer, BorderLayout.CENTER);
        panel.add(northContainer, BorderLayout.NORTH);

        JPanel centerPanel = new JPanel(new BorderLayout());

        trainingChart = new TrainingChart();
        JScrollPane chartScroll = new JScrollPane(trainingChart);
        chartScroll.setPreferredSize(new Dimension(600, 300));
        chartScroll.setBorder(BorderFactory.createTitledBorder("训练曲线"));
        centerPanel.add(chartScroll, BorderLayout.NORTH);

        centerPanel.add(trainingChart.createControlPanel(), BorderLayout.SOUTH);

        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));

        JPanel infoPanel = new JPanel(new GridLayout(6, 1, 4, 4));
        infoPanel.setBorder(BorderFactory.createTitledBorder("训练状态"));
        lossLabel = new JLabel("Loss: --");
        epochLabel = new JLabel("Epoch: --");
        accuracyLabel = new JLabel("Accuracy: --");
        crossLossLabel = new JLabel("CrossLoss: --");
        infoPanel.add(epochLabel);
        infoPanel.add(lossLabel);
        infoPanel.add(accuracyLabel);
        infoPanel.add(crossLossLabel);

        modeToggleLabel = new JLabel("<html><u>模式: 单流</u></html>");
        modeToggleLabel.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        modeToggleLabel.setFont(modeToggleLabel.getFont().deriveFont(Font.BOLD, 13f));
        modeToggleLabel.setForeground(new Color(0, 80, 180));
        modeToggleLabel.setToolTipText("点击切换 单流/双流 模式");
        modeToggleLabel.addMouseListener(new java.awt.event.MouseAdapter() {
            @Override
            public void mouseClicked(java.awt.event.MouseEvent e) {
                toggleMode();
            }
            @Override
            public void mouseEntered(java.awt.event.MouseEvent e) {
                modeToggleLabel.setForeground(new Color(0, 120, 255));
            }
            @Override
            public void mouseExited(java.awt.event.MouseEvent e) {
                modeToggleLabel.setForeground(new Color(0, 80, 180));
            }
        });
        infoPanel.add(modeToggleLabel);

        biasStatusLabel = new JLabel("偏误监测: 关");
        biasStatusLabel.setForeground(Color.GRAY);
        infoPanel.add(biasStatusLabel);

        rightPanel.add(infoPanel);
        rightPanel.add(Box.createVerticalStrut(8));

        JPanel biasPanel = new JPanel(new GridBagLayout());
        biasPanel.setBorder(BorderFactory.createTitledBorder("偏误监测"));
        biasPanel.setPreferredSize(new java.awt.Dimension(280, 140));
        biasPanel.setMinimumSize(new java.awt.Dimension(280, 120));
        biasPanel.setMaximumSize(new java.awt.Dimension(400, 200));
        GridBagConstraints bgbc = new GridBagConstraints();
        bgbc.insets = new Insets(3, 6, 3, 6);
        bgbc.anchor = GridBagConstraints.WEST;
        bgbc.fill = GridBagConstraints.HORIZONTAL;
        bgbc.gridx = 0; bgbc.gridy = 0; bgbc.gridwidth = 2;
        bgbc.weightx = 1.0;

        biasRangeSlider = new RangeSlider(0.0, 1.0, 0.0, 1.0);
        biasRangeSlider.setPreferredSize(new java.awt.Dimension(200, 30));
        biasPanel.add(biasRangeSlider, bgbc);

        bgbc.gridy = 1; bgbc.gridwidth = 1; bgbc.weightx = 0;
        bgbc.gridx = 0;
        biasPanel.add(new JLabel("范围:"), bgbc);
        bgbc.gridx = 1; bgbc.weightx = 1.0;
        biasRangeValueLabel = new JLabel("[0.000, 1.000]");
        biasRangeValueLabel.setFont(biasRangeValueLabel.getFont().deriveFont(Font.PLAIN, 11f));
        biasPanel.add(biasRangeValueLabel, bgbc);

        bgbc.gridy = 2; bgbc.gridx = 0; bgbc.weightx = 0;
        biasPanel.add(new JLabel("偏误警告:"), bgbc);
        bgbc.gridx = 1; bgbc.weightx = 1.0;
        biasWarningBox = new JComboBox<>(new String[]{"无", "仅警告", "中止"});
        biasWarningBox.addActionListener(e -> {
            int idx = biasWarningBox.getSelectedIndex();
            biasAction = idx == 0 ? BiasAction.NONE : idx == 1 ? BiasAction.WARN : BiasAction.ABORT;
            updateBiasStatus();
        });
        biasPanel.add(biasWarningBox, bgbc);

        bgbc.gridy = 3; bgbc.gridx = 0; bgbc.gridwidth = 2; bgbc.weightx = 1.0;
        JLabel biasHint = new JLabel("<html><font color='gray' size='2'>拖动红色区域设定不健康loss范围</font></html>");
        biasPanel.add(biasHint, bgbc);

        biasRangeSlider.addRangeListener((low, high) -> {
            biasLow = low;
            biasHigh = high;
            biasRangeValueLabel.setText(String.format("[%.3f, %.3f]", low, high));
            updateBiasStatus();
        });

        rightPanel.add(biasPanel);
        rightPanel.add(Box.createVerticalStrut(8));

        JPanel autoPanel = new JPanel(new GridBagLayout());
        autoPanel.setBorder(BorderFactory.createTitledBorder("安排和自动化"));
        GridBagConstraints agbc = new GridBagConstraints();
        agbc.insets = new Insets(3, 6, 3, 6);
        agbc.anchor = GridBagConstraints.WEST;
        agbc.fill = GridBagConstraints.HORIZONTAL;
        agbc.gridx = 0; agbc.gridy = 0; agbc.gridwidth = 3;

        JTextField autoFolderField = new JTextField(18);
        autoFolderField.setEditable(false);
        autoFolderField.setFont(autoFolderField.getFont().deriveFont(Font.PLAIN, 11f));
        autoPanel.add(autoFolderField, agbc);

        agbc.gridy = 1; agbc.gridwidth = 1; agbc.gridx = 0;
        JButton autoBrowseBtn = new JButton("选文件夹");
        autoPanel.add(autoBrowseBtn, agbc);
        agbc.gridx = 1;
        JComboBox<String> autoStrategyBox = new JComboBox<>(SampleExtractor.strategyLabels());
        autoStrategyBox.setFont(autoStrategyBox.getFont().deriveFont(Font.PLAIN, 11f));
        autoPanel.add(autoStrategyBox, agbc);
        agbc.gridx = 2;
        JCheckBox autoTrainBox = new JCheckBox("自动训练", false);
        autoTrainBox.setFont(autoTrainBox.getFont().deriveFont(Font.PLAIN, 11f));
        autoPanel.add(autoTrainBox, agbc);

        agbc.gridy = 2; agbc.gridx = 0; agbc.gridwidth = 2;
        JComboBox<String> autoMethodBox = new JComboBox<>(new String[]{"BAG_OF_WORDS", "TF_IDF", "CHAR_NGRAM"});
        autoMethodBox.setFont(autoMethodBox.getFont().deriveFont(Font.PLAIN, 11f));
        autoPanel.add(new JLabel("向量化:"), agbc);
        agbc.gridx = 2;
        autoPanel.add(autoMethodBox, agbc);

        agbc.gridy = 3; agbc.gridx = 0; agbc.gridwidth = 3;
        JButton autoStartBtn = new JButton("▶ 开始批量处理");
        autoStartBtn.setFont(autoStartBtn.getFont().deriveFont(Font.BOLD, 12f));
        autoStartBtn.setForeground(new Color(0, 100, 0));
        autoPanel.add(autoStartBtn, agbc);

        agbc.gridy = 4; agbc.gridwidth = 3;
        JLabel autoHint = new JLabel("<html><font color='gray' size='2'>选择文件夹后自动依次处理所有文本文件</font></html>");
        autoPanel.add(autoHint, agbc);

        autoBrowseBtn.addActionListener(e -> {
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                autoFolderField.setText(fc.getSelectedFile().getAbsolutePath());
                config.setDataPath(fc.getSelectedFile().getAbsolutePath());
            }
        });

        autoStartBtn.addActionListener(e -> runAutomation(
                autoFolderField.getText(), autoStrategyBox, autoMethodBox, autoTrainBox));

        rightPanel.add(autoPanel);
        rightPanel.add(Box.createVerticalGlue());

        centerPanel.add(rightPanel, BorderLayout.EAST);

        trainLogArea = new JTextArea();
        trainLogArea.setEditable(false);
        trainLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane sp = new JScrollPane(trainLogArea);
        sp.setBorder(BorderFactory.createTitledBorder("训练日志"));
        sp.setPreferredSize(new Dimension(0, 150));
        centerPanel.add(sp, BorderLayout.CENTER);

        panel.add(centerPanel, BorderLayout.CENTER);
        return panel;
    }

    // ========== 实例应用面板 ==========
    private JPanel buildAppPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel topBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 4));
        topBar.setBorder(BorderFactory.createTitledBorder("模型加载"));

        JButton loadModelBtn = new JButton("加载存档/Skill");
        loadModelBtn.setFont(loadModelBtn.getFont().deriveFont(Font.BOLD, 12f));
        loadModelBtn.addActionListener(e -> loadAppModel());
        topBar.add(loadModelBtn);

        JButton useCurrentBtn = new JButton("使用当前网络");
        useCurrentBtn.addActionListener(e -> useCurrentAsApp());
        topBar.add(useCurrentBtn);

        JComboBox<String> modeBox = new JComboBox<>(new String[]{"回归", "分类"});
        modeBox.addActionListener(e -> {
            appOutputMode = modeBox.getSelectedIndex() == 0 ? OutputMode.REGRESSION : OutputMode.MULTICLASS_CLASSIFICATION;
        });
        topBar.add(new JLabel("输出模式:"));
        topBar.add(modeBox);

        appModelInfoLabel = new JLabel("未加载模型");
        appModelInfoLabel.setForeground(Color.GRAY);
        appModelInfoLabel.setFont(appModelInfoLabel.getFont().deriveFont(Font.PLAIN, 11f));
        topBar.add(appModelInfoLabel);

        panel.add(topBar, BorderLayout.NORTH);

        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        mainSplit.setResizeWeight(0.45);

        JPanel leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS));

        appInputArea = new JTextArea(6, 30);
        appInputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        appInputArea.setLineWrap(true);
        appInputArea.setWrapStyleWord(true);
        JScrollPane inputScroll = new JScrollPane(appInputArea);
        inputScroll.setBorder(BorderFactory.createTitledBorder("输入"));
        inputScroll.setPreferredSize(new Dimension(0, 120));
        leftPanel.add(inputScroll);

        JPanel inputBtnPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        JButton runBtn = new JButton("▶ 运行推理");
        runBtn.setFont(runBtn.getFont().deriveFont(Font.BOLD, 13f));
        runBtn.setForeground(new Color(0, 100, 0));
        runBtn.addActionListener(e -> runAppInference());
        inputBtnPanel.add(runBtn);

        JButton clearInputBtn = new JButton("清空输入");
        clearInputBtn.addActionListener(e -> appInputArea.setText(""));
        inputBtnPanel.add(clearInputBtn);

        JButton clearAllBtn = new JButton("清空历史");
        clearAllBtn.setForeground(new Color(180, 0, 0));
        clearAllBtn.addActionListener(e -> {
            appHistoryArea.setText("");
            appOutputArea.setText("");
            appThinkingArea.setText("");
            appStatusLabel.setText(" ");
        });
        inputBtnPanel.add(clearAllBtn);

        appStatusLabel = new JLabel(" ");
        appStatusLabel.setForeground(Color.GRAY);
        inputBtnPanel.add(appStatusLabel);

        leftPanel.add(inputBtnPanel);

        JPanel genPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        genPanel.setBorder(BorderFactory.createTitledBorder("模型自主生成"));

        JButton buildGenBtn = new JButton("构建生成器");
        buildGenBtn.addActionListener(e -> buildTextGenerator());
        genPanel.add(buildGenBtn);

        JButton autoGenBtn = new JButton("自主生成");
        autoGenBtn.setFont(autoGenBtn.getFont().deriveFont(Font.BOLD, 12f));
        autoGenBtn.setForeground(new Color(128, 0, 128));
        autoGenBtn.addActionListener(e -> runAutonomousGeneration());
        genPanel.add(autoGenBtn);

        genPanel.add(new JLabel("温度:"));
        JTextField genTempField = new JTextField("0.8", 3);
        genPanel.add(genTempField);

        genPanel.add(new JLabel("长度:"));
        JTextField genLenField = new JTextField("100", 4);
        genPanel.add(genLenField);

        JCheckBox wordLevelBox = new JCheckBox("词级", true);
        genPanel.add(wordLevelBox);

        preservePunctuationBox = new JCheckBox("保留标点/序号", false);
        preservePunctuationBox.setToolTipText("生成结果中保留标点符号、序号等结构元素");
        preservePunctuationBox.setForeground(new Color(0, 100, 0));
        genPanel.add(preservePunctuationBox);

        JLabel genInfoLabel = new JLabel("未构建");
        genInfoLabel.setForeground(Color.GRAY);
        genInfoLabel.setFont(genInfoLabel.getFont().deriveFont(Font.PLAIN, 10f));
        genPanel.add(genInfoLabel);

        leftPanel.add(genPanel);

        JPanel apiPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        apiPanel.setBorder(BorderFactory.createTitledBorder("推理API服务"));

        apiPortField = new JTextField("8891", 4);
        apiPanel.add(new JLabel("端口:"));
        apiPanel.add(apiPortField);

        apiToggleBtn = new JButton("启动API");
        apiToggleBtn.setForeground(new Color(0, 100, 0));
        apiToggleBtn.setFont(apiToggleBtn.getFont().deriveFont(Font.BOLD, 11f));
        apiToggleBtn.addActionListener(e -> toggleApiServer());
        apiPanel.add(apiToggleBtn);

        apiStatusLabel = new JLabel("未启动");
        apiStatusLabel.setForeground(Color.GRAY);
        apiStatusLabel.setFont(apiStatusLabel.getFont().deriveFont(Font.PLAIN, 10f));
        apiPanel.add(apiStatusLabel);

        leftPanel.add(apiPanel);

        appHistoryArea = new JTextArea();
        appHistoryArea.setEditable(false);
        appHistoryArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        appHistoryArea.setLineWrap(true);
        appHistoryArea.setWrapStyleWord(true);
        JScrollPane historyScroll = new JScrollPane(appHistoryArea);
        historyScroll.setBorder(BorderFactory.createTitledBorder("交互历史"));
        leftPanel.add(historyScroll);

        mainSplit.setLeftComponent(leftPanel);

        JPanel rightPanel = new JPanel(new BorderLayout(4, 4));

        appOutputArea = new JTextArea();
        appOutputArea.setEditable(false);
        appOutputArea.setFont(new Font(Font.MONOSPACED, Font.BOLD, 13));
        appOutputArea.setLineWrap(true);
        appOutputArea.setWrapStyleWord(true);
        appOutputArea.setBackground(new Color(245, 250, 245));
        JScrollPane outputScroll = new JScrollPane(appOutputArea);
        outputScroll.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createCompoundBorder(
                        BorderFactory.createLineBorder(new Color(0, 120, 0)),
                        BorderFactory.createEmptyBorder(2, 4, 2, 4)),
                "输出结果"));
        outputScroll.setPreferredSize(new Dimension(0, 150));
        rightPanel.add(outputScroll, BorderLayout.NORTH);

        appThinkingArea = new JTextArea();
        appThinkingArea.setEditable(false);
        appThinkingArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        appThinkingArea.setLineWrap(true);
        appThinkingArea.setWrapStyleWord(true);
        appThinkingArea.setBackground(new Color(248, 248, 255));
        JScrollPane thinkScroll = new JScrollPane(appThinkingArea);
        thinkScroll.setBorder(BorderFactory.createTitledBorder("思考过程 (逐层分析)"));
        rightPanel.add(thinkScroll, BorderLayout.CENTER);

        mainSplit.setRightComponent(rightPanel);

        panel.add(mainSplit, BorderLayout.CENTER);

        return panel;
    }

    private void loadAppModel() {
        JFileChooser fc = new JFileChooser(saveManager.getSaveDir());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("MaChain存档", "mcsave", "mcskill"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            SaveManager.SaveData sd = saveManager.load(fc.getSelectedFile().getAbsolutePath());
            appNetwork = sd.network;
            appModelInfoLabel.setText("已加载: " + fc.getSelectedFile().getName()
                    + " | " + sd.type + " | " + appNetwork.getArchitecture()
                    + " | 参数: " + appNetwork.countParameters()
                    + " | 已训练: " + appNetwork.getTotalEpochs() + "轮");
            appModelInfoLabel.setForeground(new Color(0, 100, 0));
            appStatusLabel.setText("模型就绪");
            appHistoryArea.append("[系统] 已加载模型: " + fc.getSelectedFile().getName()
                    + " (" + sd.type + ", " + sd.note + ")\n");
        } catch (Exception ex) {
            appStatusLabel.setText("加载失败: " + ex.getMessage());
        }
    }

    private void useCurrentAsApp() {
        if (dualMode && dualNetwork != null) {
            appNetwork = dualNetwork.getTextNetwork();
            appTextProcessor = structTextProcessor;
            appModelInfoLabel.setText("当前双流网络(文本子网) | " + appNetwork.getArchitecture());
        } else if (network != null) {
            appNetwork = network;
            appTextProcessor = textProcessor != null ? textProcessor : this.textProcessor;
            appModelInfoLabel.setText("当前单流网络 | " + appNetwork.getArchitecture()
                    + " | 训练: " + appNetwork.getTotalEpochs() + "轮");
        } else {
            appStatusLabel.setText("当前没有可用的网络");
            return;
        }
        appModelInfoLabel.setForeground(new Color(0, 100, 0));
        appStatusLabel.setText("已切换为当前网络");
        appHistoryArea.append("[系统] 已加载当前网络: " + appNetwork.getArchitecture() + "\n");
    }

    private void runAppInference() {
        if (appNetwork == null) {
            appStatusLabel.setText("请先加载模型");
            return;
        }
        String text = appInputArea.getText().trim();
        if (text.isEmpty()) {
            appStatusLabel.setText("请输入内容");
            return;
        }

        try {
            double[] input;
            int expectedDim = appNetwork.getLayerSizes()[0];

            if (appTextProcessor != null && !text.matches("[\\d.,;\\s]+")) {
                input = appTextProcessor.vectorize(text);
                if (input.length != expectedDim) {
                    double[] padded = new double[expectedDim];
                    System.arraycopy(input, 0, padded, 0, Math.min(input.length, expectedDim));
                    input = padded;
                }
            } else {
                String[] parts = text.split("[,;\\s]+");
                input = new double[expectedDim];
                for (int i = 0; i < Math.min(parts.length, expectedDim); i++) {
                    try { input[i] = Double.parseDouble(parts[i].trim()); } catch (Exception ignored) {}
                }
            }

            StringBuilder thinking = new StringBuilder();
            thinking.append("═══ 推理开始 ═══\n");
            thinking.append("模型: ").append(appNetwork.getArchitecture()).append("\n");
            thinking.append("参数量: ").append(appNetwork.countParameters()).append("\n");
            thinking.append("输入维度: ").append(input.length).append("\n");
            thinking.append("输出模式: ").append(appOutputMode.getDisplayName()).append("\n\n");

            double[] current = input.clone();
            int[] sizes = appNetwork.getLayerSizes();
            Network.Activation[] acts = appNetwork.getLayerActivations();
            double[][] weights = appNetwork.getWeights();
            double[][] biases = appNetwork.getBiases();

            for (int l = 0; l < acts.length; l++) {
                int fanIn = sizes[l];
                int fanOut = sizes[l + 1];
                double[] next = new double[fanOut];
                double sumAbs = 0;
                double maxVal = Double.NEGATIVE_INFINITY;
                double minVal = Double.POSITIVE_INFINITY;
                int activeCount = 0;

                for (int j = 0; j < fanOut; j++) {
                    double s = biases[l][j];
                    for (int i = 0; i < fanIn; i++) {
                        s += current[i] * weights[l][i * fanOut + j];
                    }
                    double activated;
                    switch (acts[l]) {
                        case SIGMOID: activated = 1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, s)))); break;
                        case RELU: activated = Math.max(0, s); break;
                        case TANH: activated = Math.tanh(s); break;
                        default: activated = s;
                    }
                    next[j] = activated;
                    sumAbs += Math.abs(activated);
                    maxVal = Math.max(maxVal, activated);
                    minVal = Math.min(minVal, activated);
                    if (activated > 0.01) activeCount++;
                }

                thinking.append("第").append(l + 1).append("层 [")
                        .append(fanIn).append("→").append(fanOut).append("] ")
                        .append(acts[l]).append("\n");
                thinking.append("  均值=").append(String.format("%.4f", sumAbs / fanOut));
                thinking.append(" 最大=").append(String.format("%.4f", maxVal));
                thinking.append(" 最小=").append(String.format("%.4f", minVal));
                thinking.append(" 活跃=").append(activeCount).append("/").append(fanOut);
                thinking.append(" (").append(String.format("%.0f%%", 100.0 * activeCount / fanOut)).append(")\n");

                double topN = Math.min(5, fanOut);
                double[] sorted = next.clone();
                Arrays.sort(sorted);
                thinking.append("  Top").append((int) topN).append(": ");
                for (int i = sorted.length - 1; i >= Math.max(0, sorted.length - (int) topN); i--) {
                    thinking.append(String.format("%.4f", sorted[i]));
                    if (i > Math.max(0, sorted.length - (int) topN)) thinking.append(", ");
                }
                thinking.append("\n\n");

                current = next;
            }

            StringBuilder outputStr = new StringBuilder();
            thinking.append("═══ 最终输出 ═══\n");

            if (appOutputMode == OutputMode.REGRESSION) {
                for (int i = 0; i < current.length; i++) {
                    if (i > 0) outputStr.append(", ");
                    outputStr.append(String.format("%.6f", current[i]));
                }
                thinking.append("回归输出: ").append(outputStr).append("\n");
            } else {
                double sum = 0;
                for (double v : current) sum += v;
                double maxV = Double.NEGATIVE_INFINITY;
                int bestIdx = 0;
                for (int i = 0; i < current.length; i++) {
                    if (current[i] > maxV) { maxV = current[i]; bestIdx = i; }
                }
                outputStr.append("预测类别: ").append(bestIdx);
                if (appLabelNames != null && bestIdx < appLabelNames.length) {
                    outputStr.append(" (").append(appLabelNames[bestIdx]).append(")");
                }
                outputStr.append("\n置信度: ").append(String.format("%.2f%%",
                        sum > 0 ? 100.0 * maxV / sum : 0));
                outputStr.append("\n\n各类别概率:\n");
                for (int i = 0; i < current.length; i++) {
                    double prob = sum > 0 ? current[i] / sum : 0;
                    outputStr.append(String.format("  [%d] %.4f", i, prob));
                    if (appLabelNames != null && i < appLabelNames.length) {
                        outputStr.append(" ").append(appLabelNames[i]);
                    }
                    int barLen = (int) (prob * 20);
                    outputStr.append(" ").append("█".repeat(Math.max(0, barLen)));
                    outputStr.append("\n");
                }
                thinking.append("分类结果: 类别 ").append(bestIdx);
                if (appLabelNames != null && bestIdx < appLabelNames.length) {
                    thinking.append(" (").append(appLabelNames[bestIdx]).append(")");
                }
                thinking.append("\n置信度: ").append(String.format("%.2f%%",
                        sum > 0 ? 100.0 * maxV / sum : 0)).append("\n");
            }

            appOutputArea.setText(outputStr.toString());
            appThinkingArea.setText(thinking.toString());
            appStatusLabel.setText("推理完成");

            appHistoryArea.append("\n[输入] " +
                    (text.length() > 60 ? text.substring(0, 60) + "..." : text) + "\n");
            appHistoryArea.append("[输出] " +
                    outputStr.toString().replace("\n", " | ") + "\n");
            appHistoryArea.setCaretPosition(appHistoryArea.getDocument().getLength());

        } catch (Exception ex) {
            appOutputArea.setText("推理失败");
            appThinkingArea.setText("错误: " + ex.getMessage());
            appStatusLabel.setText("推理出错");
        }
    }

    private void buildTextGenerator() {
        machain.data.DataSet ds = textDataSet != null ? textDataSet : (dataSet != null ? dataSet : null);
        if (ds == null && originalTexts.isEmpty()) {
            appStatusLabel.setText("请先加载文本数据");
            return;
        }

        Network net = appNetwork != null ? appNetwork : (network != null ? network : null);
        TextProcessor tp = appTextProcessor != null ? appTextProcessor : (textProcessor != null ? textProcessor : null);

        List<String> texts = new ArrayList<>();

        if (!originalTexts.isEmpty()) {
            texts.addAll(originalTexts);
        } else if (ds != null) {
            String[] labelNames = ds.getLabelNames();
            if (labelNames != null && labelNames.length > 0) {
                for (String label : labelNames) {
                    texts.add(label);
                }
            }
        }

        if (texts.isEmpty()) {
            appStatusLabel.setText("没有可用的文本数据");
            return;
        }

        textGenerator = new TextGenerator();
        boolean preservePunct = preservePunctuationBox != null && preservePunctuationBox.isSelected();
        textGenerator.setPreservePunctuation(preservePunct);
        if (tp != null) {
            tp.setPreservePunctuation(preservePunct);
        }
        textGenerator.textProcessor = tp;
        textGenerator.network = net;
        textGenerator.enhancedNetwork = enhancedNetwork;
        textGenerator.buildFromTexts(texts, true);

        appStatusLabel.setText("生成器已构建: " + textGenerator.getStats());
        appHistoryArea.append("[生成器] 已构建: " + textGenerator.getStats() + "\n");
        syncApiServer();
    }

    private void syncApiServer() {
        if (apiServer == null || !apiServer.isRunning()) return;
        Network net = appNetwork != null ? appNetwork : network;
        TextProcessor tp = appTextProcessor != null ? appTextProcessor : textProcessor;
        apiServer.setModel(net, tp, appOutputMode);
        apiServer.setTextGenerator(textGenerator);
        apiServer.setEnhancedNetwork(enhancedNetwork);
    }

    private void runAutonomousGeneration() {
        if (textGenerator == null) {
            buildTextGenerator();
            if (textGenerator == null) return;
        }

        String seed = appInputArea.getText().trim();
        double temp = 0.8;
        int maxLen = 100;
        try {
            Component[] comps = ((JPanel) ((JPanel) appInputArea.getParent().getParent().getParent())
                    .getComponent(2)).getComponents();
            for (Component c : comps) {
                if (c instanceof JPanel gp) {
                    for (Component child : gp.getComponents()) {
                        if (child instanceof JTextField tf) {
                            if (tf.getText().contains(".")) {
                                temp = Double.parseDouble(tf.getText().trim());
                            } else {
                                maxLen = Integer.parseInt(tf.getText().trim());
                            }
                        }
                    }
                }
            }
        } catch (Exception ignored) {}

        textGenerator.setTemperature(temp);
        textGenerator.setMaxGenLength(maxLen);

        StringBuilder thinking = new StringBuilder();
        thinking.append("═══ 模型自主生成 ═══\n");
        thinking.append("生成器: ").append(textGenerator.getStats()).append("\n");
        thinking.append("种子输入: ").append(seed.isEmpty() ? "(无，自由生成)" : seed).append("\n");
        thinking.append("温度: ").append(String.format("%.2f", temp)).append("\n");
        thinking.append("最大长度: ").append(maxLen).append("\n\n");

        long start = System.currentTimeMillis();
        String generated = textGenerator.generate(seed.isEmpty() ? null : seed);
        long elapsed = System.currentTimeMillis() - start;

        thinking.append("生成耗时: ").append(elapsed).append("ms\n");
        thinking.append("输出长度: ").append(generated.length()).append(" 字符\n\n");
        thinking.append("═══ 生成过程 ═══\n");
        thinking.append("模型基于训练数据的统计规律和已学习到的隐层表示\n");
        thinking.append("自主决定每一个输出token。\n");
        thinking.append("网络偏置权重: ").append(String.format("%.2f", textGenerator.getNetworkBiasWeight())).append("\n");
        if (appNetwork != null) {
            thinking.append("使用网络: ").append(appNetwork.getArchitecture()).append("\n");
            thinking.append("隐层维度: ");
            int[] sizes = appNetwork.getLayerSizes();
            for (int i = 1; i < sizes.length - 1; i++) {
                if (i > 1) thinking.append(", ");
                thinking.append(sizes[i]);
            }
            thinking.append("\n");
        }

        appOutputArea.setText(generated);
        appThinkingArea.setText(thinking.toString());
        appStatusLabel.setText(String.format("自主生成完成 | %d字 | %dms", generated.length(), elapsed));

        appHistoryArea.append("[自主生成] " +
                (seed.length() > 30 ? seed.substring(0, 30) + "..." : seed) +
                " → " + generated.length() + "字\n");
        appHistoryArea.setCaretPosition(appHistoryArea.getDocument().getLength());
    }

    // ========== 接口中心面板 ==========
    private JPanel buildInterfacePanel() {
        sampleScraper = new SampleScraper();
        sampleScraper.setListener(new SampleScraper.ScraperListener() {
            @Override
            public void onSampleCaptured(String source, String text) {
                SwingUtilities.invokeLater(() -> {
                    interfaceLogArea.append("[抓取] " + source + " (" + text.length() + "字)\n");
                    interfaceLogArea.setCaretPosition(interfaceLogArea.getDocument().getLength());
                });
            }
            @Override
            public void onError(String source, String message) {
                SwingUtilities.invokeLater(() ->
                        interfaceLogArea.append("[错误] " + source + ": " + message + "\n"));
            }
        });

        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JTabbedPane subTabs = new JTabbedPane(JTabbedPane.TOP);
        subTabs.addTab("输入接口", buildInputInterfacePanel());
        subTabs.addTab("样本抓取", buildScrapePanel());
        subTabs.addTab("输出接口", buildOutputInterfacePanel());

        panel.add(subTabs, BorderLayout.CENTER);

        interfaceLogArea = new JTextArea(6, 40);
        interfaceLogArea.setEditable(false);
        interfaceLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        JScrollPane logScroll = new JScrollPane(interfaceLogArea);
        logScroll.setBorder(BorderFactory.createTitledBorder("接口日志"));
        logScroll.setPreferredSize(new Dimension(0, 120));
        panel.add(logScroll, BorderLayout.SOUTH);

        return panel;
    }

    private JPanel buildInputInterfacePanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));

        JPanel topPanel = new JPanel(new BorderLayout(4, 4));
        topPanel.setBorder(BorderFactory.createTitledBorder("数据输入"));

        JTextArea formInputArea = new JTextArea(8, 40);
        formInputArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        formInputArea.setLineWrap(true);
        JScrollPane formScroll = new JScrollPane(formInputArea);
        formScroll.setBorder(BorderFactory.createTitledBorder(
                "批量输入 (每行一条，Tab分隔输入和标签)"));
        topPanel.add(formScroll, BorderLayout.CENTER);

        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));

        JButton importFormBtn = new JButton("导入到数据集");
        importFormBtn.addActionListener(e -> {
            String text = formInputArea.getText().trim();
            if (text.isEmpty()) return;
            List<SampleExtractor.ExtractedSample> samples = new ArrayList<>();
            for (String line : text.split("\n")) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split("\t", 2);
                String input = parts[0].trim();
                String label = parts.length > 1 ? parts[1].trim() : "未知";
                samples.add(new SampleExtractor.ExtractedSample(input, label));
            }
            if (samples.isEmpty()) return;
            try {
                TextProcessor tp = new TextProcessor(TextProcessor.VectorizeMethod.TF_IDF, 1000);
                SampleExtractor helper = new SampleExtractor(SampleExtractor.Strategy.ARTICLE_TOPIC);
                machain.data.DataSet ds = helper.toDataSet(samples, tp);
                if (dualMode) {
                    textDataSet = ds;
                    structTextProcessor = tp;
                    textLayerField.setText(tp.getVectorSize() + ",32," + ds.outputDim());
                    textDataLabel.setText("接口导入: " + samples.size() + " 条");
                } else {
                    dataSet = ds;
                    textProcessor = tp;
                    layerField.setText(tp.getVectorSize() + ",32," + ds.outputDim());
                    dataStatusLabel.setText("接口导入: " + samples.size() + " 条");
                }
                interfaceLogArea.append("[输入] 导入 " + samples.size() + " 条样本到"
                        + (dualMode ? "文本通道" : "单流数据集") + "\n");
            } catch (Exception ex) {
                interfaceLogArea.append("[输入错误] " + ex.getMessage() + "\n");
            }
        });
        btnPanel.add(importFormBtn);

        JButton pasteClipBtn = new JButton("从剪贴板粘贴");
        pasteClipBtn.addActionListener(e -> {
            String clip = sampleScraper.getClipboardText();
            if (!clip.isEmpty()) {
                formInputArea.setText(clip);
                interfaceLogArea.append("[输入] 从剪贴板粘贴 " + clip.length() + " 字\n");
            }
        });
        btnPanel.add(pasteClipBtn);

        JButton loadFileBtn = new JButton("从文件加载");
        loadFileBtn.addActionListener(e -> {
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                    "文本文件", "txt", "tsv", "csv", "md"));
            fc.setMultiSelectionEnabled(true);
            if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
            StringBuilder sb = new StringBuilder();
            for (File f : fc.getSelectedFiles()) {
                try {
                    String content = java.nio.file.Files.readString(f.toPath(),
                            java.nio.charset.StandardCharsets.UTF_8);
                    sb.append(content).append("\n");
                    interfaceLogArea.append("[输入] 加载文件: " + f.getName()
                            + " (" + content.length() + "字)\n");
                } catch (Exception ex) {
                    interfaceLogArea.append("[输入错误] " + f.getName() + ": " + ex.getMessage() + "\n");
                }
            }
            formInputArea.setText(sb.toString());
        });
        btnPanel.add(loadFileBtn);

        topPanel.add(btnPanel, BorderLayout.SOUTH);
        panel.add(topPanel, BorderLayout.CENTER);

        return panel;
    }

    private JPanel buildScrapePanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));

        JPanel urlPanel = new JPanel(new BorderLayout(4, 4));
        urlPanel.setBorder(BorderFactory.createTitledBorder("URL抓取"));

        JPanel urlInputPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        urlInputPanel.add(new JLabel("URL:"));
        urlInputField = new JTextField(40);
        urlInputField.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        urlInputPanel.add(urlInputField);

        JButton fetchBtn = new JButton("抓取");
        fetchBtn.addActionListener(e -> {
            String url = urlInputField.getText().trim();
            if (url.isEmpty()) return;
            interfaceLogArea.append("[抓取] 正在获取: " + url + "\n");
            new Thread(() -> {
                String text = sampleScraper.fetchUrl(url);
                SwingUtilities.invokeLater(() -> {
                    scrapedTextArea.setText(text);
                    interfaceLogArea.append("[抓取] 完成: " + text.length() + " 字\n");
                });
            }).start();
        });
        urlInputPanel.add(fetchBtn);

        JButton batchFetchBtn = new JButton("批量抓取");
        batchFetchBtn.addActionListener(e -> {
            String text = urlInputField.getText().trim();
            if (text.isEmpty()) return;
            String[] urls = text.split("\\s*[;\\n]\\s*");
            interfaceLogArea.append("[抓取] 批量抓取 " + urls.length + " 个URL\n");
            new Thread(() -> {
                List<String> results = sampleScraper.fetchMultipleUrls(Arrays.asList(urls));
                SwingUtilities.invokeLater(() -> {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < results.size(); i++) {
                        sb.append("=== URL ").append(i + 1).append(" ===\n");
                        sb.append(results.get(i)).append("\n\n");
                    }
                    scrapedTextArea.setText(sb.toString());
                    interfaceLogArea.append("[抓取] 批量完成: " + results.size() + " 个结果\n");
                });
            }).start();
        });
        urlInputPanel.add(batchFetchBtn);
        urlPanel.add(urlInputPanel, BorderLayout.NORTH);

        scrapedTextArea = new JTextArea();
        scrapedTextArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        scrapedTextArea.setLineWrap(true);
        scrapedTextArea.setWrapStyleWord(true);
        JScrollPane scrapeScroll = new JScrollPane(scrapedTextArea);
        scrapeScroll.setBorder(BorderFactory.createTitledBorder("抓取结果"));
        urlPanel.add(scrapeScroll, BorderLayout.CENTER);

        JPanel scrapeBtnPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        JButton useScrapedBtn = new JButton("用作训练数据");
        useScrapedBtn.addActionListener(e -> {
            String text = scrapedTextArea.getText().trim();
            if (text.isEmpty()) return;
            try {
                List<String> paragraphs = SampleScraper.splitIntoParagraphs(text);
                List<SampleExtractor.ExtractedSample> samples = new ArrayList<>();
                SampleExtractor extractor = new SampleExtractor(SampleExtractor.Strategy.SEQUENCE_PREDICT);
                samples.addAll(extractor.extract(Collections.singletonList(text)));
                if (samples.isEmpty()) {
                    for (String p : paragraphs) {
                        samples.add(new SampleExtractor.ExtractedSample(p, "段落"));
                    }
                }
                TextProcessor tp = new TextProcessor(TextProcessor.VectorizeMethod.TF_IDF, 1000);
                SampleExtractor helper = new SampleExtractor(SampleExtractor.Strategy.ARTICLE_TOPIC);
                machain.data.DataSet ds = helper.toDataSet(samples, tp);
                if (dualMode) {
                    textDataSet = ds;
                    structTextProcessor = tp;
                    textLayerField.setText(tp.getVectorSize() + ",32," + ds.outputDim());
                    textDataLabel.setText("URL抓取: " + samples.size() + " 条");
                } else {
                    dataSet = ds;
                    textProcessor = tp;
                    layerField.setText(tp.getVectorSize() + ",32," + ds.outputDim());
                    dataStatusLabel.setText("URL抓取: " + samples.size() + " 条");
                }
                interfaceLogArea.append("[抓取] " + samples.size() + " 条样本已导入数据集\n");
            } catch (Exception ex) {
                interfaceLogArea.append("[抓取错误] " + ex.getMessage() + "\n");
            }
        });
        scrapeBtnPanel.add(useScrapedBtn);

        JButton saveScrapedBtn = new JButton("保存为文件");
        saveScrapedBtn.addActionListener(e -> {
            String text = scrapedTextArea.getText().trim();
            if (text.isEmpty()) return;
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setSelectedFile(new File("scraped_text.txt"));
            if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                try {
                    java.nio.file.Files.writeString(fc.getSelectedFile().toPath(), text,
                            java.nio.charset.StandardCharsets.UTF_8);
                    interfaceLogArea.append("[抓取] 已保存: " + fc.getSelectedFile().getName() + "\n");
                } catch (Exception ex) {
                    interfaceLogArea.append("[保存错误] " + ex.getMessage() + "\n");
                }
            }
        });
        scrapeBtnPanel.add(saveScrapedBtn);
        urlPanel.add(scrapeBtnPanel, BorderLayout.SOUTH);

        panel.add(urlPanel, BorderLayout.CENTER);

        JPanel monitorPanel = new JPanel(new GridBagLayout());
        monitorPanel.setBorder(BorderFactory.createTitledBorder("实时监控"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 6, 4, 6);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 3;

        JButton clipMonBtn = new JButton("启动剪贴板监控");
        clipboardStatusLabel = new JLabel("未启动");
        clipboardStatusLabel.setForeground(Color.GRAY);
        clipMonBtn.addActionListener(e -> {
            if (sampleScraper.isMonitoring()) {
                sampleScraper.stopClipboardMonitor();
                clipMonBtn.setText("启动剪贴板监控");
                clipboardStatusLabel.setText("已停止");
                clipboardStatusLabel.setForeground(Color.GRAY);
                interfaceLogArea.append("[监控] 剪贴板监控已停止\n");
            } else {
                sampleScraper.startClipboardMonitor(1000);
                clipMonBtn.setText("停止剪贴板监控");
                clipboardStatusLabel.setText("监控中...");
                clipboardStatusLabel.setForeground(new Color(0, 120, 0));
                interfaceLogArea.append("[监控] 剪贴板监控已启动 (1秒间隔)\n");
            }
        });
        monitorPanel.add(clipMonBtn, gbc);

        gbc.gridy = 1; gbc.gridwidth = 1; gbc.gridx = 0;
        monitorPanel.add(clipboardStatusLabel, gbc);

        gbc.gridy = 2; gbc.gridx = 0; gbc.gridwidth = 3;
        JPanel folderPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        JTextField folderWatchField = new JTextField(25);
        folderWatchField.setEditable(false);
        folderPanel.add(folderWatchField);
        JButton folderWatchBtn = new JButton("选择监控文件夹");
        folderWatchBtn.addActionListener(e -> {
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                String path = fc.getSelectedFile().getAbsolutePath();
                folderWatchField.setText(path);
                sampleScraper.startFolderWatch(path);
                folderWatchStatusLabel.setText("监控中: " + path);
                folderWatchStatusLabel.setForeground(new Color(0, 120, 0));
                interfaceLogArea.append("[监控] 文件夹监控已启动: " + path + "\n");
            }
        });
        folderPanel.add(folderWatchBtn);
        monitorPanel.add(folderPanel, gbc);

        gbc.gridy = 3;
        folderWatchStatusLabel = new JLabel("未启动");
        folderWatchStatusLabel.setForeground(Color.GRAY);
        monitorPanel.add(folderWatchStatusLabel, gbc);

        panel.add(monitorPanel, BorderLayout.SOUTH);

        return panel;
    }

    private JPanel buildOutputInterfacePanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));

        JPanel exportPanel = new JPanel(new BorderLayout(4, 4));
        exportPanel.setBorder(BorderFactory.createTitledBorder("数据导出"));

        JPanel formatPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        formatPanel.add(new JLabel("导出格式:"));
        JComboBox<String> formatBox = new JComboBox<>(new String[]{"CSV", "JSON", "TSV", "TXT"});
        formatPanel.add(formatBox);

        JComboBox<String> dataSourceBox = new JComboBox<>(new String[]{
                "当前训练数据", "当前测试数据", "结构化通道数据", "文本通道数据", "推理结果"});
        formatPanel.add(new JLabel("数据源:"));
        formatPanel.add(dataSourceBox);

        JButton exportBtn = new JButton("导出");
        exportBtn.setFont(exportBtn.getFont().deriveFont(Font.BOLD));
        exportBtn.addActionListener(e -> {
            String format = formatBox.getSelectedItem().toString();
            int srcIdx = dataSourceBox.getSelectedIndex();
            machain.data.DataSet ds = null;
            String baseName = "export";
            switch (srcIdx) {
                case 0: ds = trainSet != null ? trainSet : dataSet; baseName = "train_data"; break;
                case 1: ds = testSet; baseName = "test_data"; break;
                case 2: ds = structDataSet; baseName = "struct_data"; break;
                case 3: ds = textDataSet; baseName = "text_data"; break;
            }

            if (ds == null) {
                interfaceLogArea.append("[输出] 所选数据源为空\n");
                return;
            }

            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setSelectedFile(new File(baseName + "." + format.toLowerCase()));
            if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;

            try {
                String path = fc.getSelectedFile().getAbsolutePath();
                String[] headers = ds.getFeatureNames();
                List<double[]> rows = new ArrayList<>();
                double[][] inputs = ds.getInputs();
                double[][] targets = ds.getTargets();
                for (int i = 0; i < inputs.length; i++) {
                    double[] combined = new double[inputs[i].length + targets[i].length];
                    System.arraycopy(inputs[i], 0, combined, 0, inputs[i].length);
                    System.arraycopy(targets[i], 0, combined, inputs[i].length, targets[i].length);
                    rows.add(combined);
                }
                String[] allHeaders = new String[headers.length + ds.getLabelNames().length];
                System.arraycopy(headers, 0, allHeaders, 0, headers.length);
                System.arraycopy(ds.getLabelNames(), 0, allHeaders, headers.length, ds.getLabelNames().length);

                switch (format) {
                    case "CSV": OutputExporter.exportCSV(path, allHeaders, rows); break;
                    case "JSON": OutputExporter.exportJSON(path, allHeaders, rows); break;
                    case "TSV": OutputExporter.exportTSV(path, allHeaders, rows); break;
                    case "TXT":
                        List<String> lines = new ArrayList<>();
                        for (double[] row : rows) {
                            StringBuilder sb = new StringBuilder();
                            for (double v : row) sb.append(String.format("%.6f ", v));
                            lines.add(sb.toString().trim());
                        }
                        OutputExporter.exportText(path, lines);
                        break;
                }
                interfaceLogArea.append("[输出] 已导出 " + rows.size() + " 条到 " + path + "\n");
            } catch (Exception ex) {
                interfaceLogArea.append("[输出错误] " + ex.getMessage() + "\n");
            }
        });
        formatPanel.add(exportBtn);

        JButton batchExportBtn = new JButton("批量导出(CSV+JSON)");
        batchExportBtn.addActionListener(e -> {
            machain.data.DataSet ds = trainSet != null ? trainSet : dataSet;
            if (ds == null) { interfaceLogArea.append("[输出] 无数据可导出\n"); return; }
            JFileChooser fc = new JFileChooser(config.getDataPath().isEmpty() ? "." : config.getDataPath());
            fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
            try {
                List<double[]> rows = new ArrayList<>();
                double[][] inputs = ds.getInputs();
                double[][] targets = ds.getTargets();
                for (int i = 0; i < inputs.length; i++) {
                    double[] combined = new double[inputs[i].length + targets[i].length];
                    System.arraycopy(inputs[i], 0, combined, 0, inputs[i].length);
                    System.arraycopy(targets[i], 0, combined, inputs[i].length, targets[i].length);
                    rows.add(combined);
                }
                String[] allHeaders = new String[ds.getFeatureNames().length + ds.getLabelNames().length];
                System.arraycopy(ds.getFeatureNames(), 0, allHeaders, 0, ds.getFeatureNames().length);
                System.arraycopy(ds.getLabelNames(), 0, allHeaders,
                        ds.getFeatureNames().length, ds.getLabelNames().length);
                OutputExporter.exportBatch(fc.getSelectedFile().getAbsolutePath(),
                        "machain_export", allHeaders, rows, "CSV", "JSON");
                interfaceLogArea.append("[输出] 批量导出完成: CSV + JSON\n");
            } catch (Exception ex) {
                interfaceLogArea.append("[输出错误] " + ex.getMessage() + "\n");
            }
        });
        formatPanel.add(batchExportBtn);

        exportPanel.add(formatPanel, BorderLayout.NORTH);

        JTextArea previewArea = new JTextArea(10, 40);
        previewArea.setEditable(false);
        previewArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        JScrollPane previewScroll = new JScrollPane(previewArea);
        previewScroll.setBorder(BorderFactory.createTitledBorder("数据预览"));
        exportPanel.add(previewScroll, BorderLayout.CENTER);

        JPanel previewBtnPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 2));
        JButton refreshPreviewBtn = new JButton("刷新预览");
        refreshPreviewBtn.addActionListener(e -> {
            int srcIdx = dataSourceBox.getSelectedIndex();
            machain.data.DataSet ds = null;
            switch (srcIdx) {
                case 0: ds = trainSet != null ? trainSet : dataSet; break;
                case 1: ds = testSet; break;
                case 2: ds = structDataSet; break;
                case 3: ds = textDataSet; break;
            }
            if (ds != null) {
                previewArea.setText(ds.getAnalysis() + "\n\n" + ds.preview(30));
            } else {
                previewArea.setText("所选数据源为空");
            }
        });
        previewBtnPanel.add(refreshPreviewBtn);

        JButton copyClipBtn = new JButton("复制到剪贴板");
        copyClipBtn.addActionListener(e -> {
            String text = previewArea.getText();
            if (!text.isEmpty()) {
                java.awt.datatransfer.StringSelection sel =
                        new java.awt.datatransfer.StringSelection(text);
                Toolkit.getDefaultToolkit().getSystemClipboard().setContents(sel, null);
                interfaceLogArea.append("[输出] 已复制到剪贴板\n");
            }
        });
        previewBtnPanel.add(copyClipBtn);
        exportPanel.add(previewBtnPanel, BorderLayout.SOUTH);

        panel.add(exportPanel, BorderLayout.CENTER);

        return panel;
    }

    private void applyRegularization(Network net) {
        double baseWeightDecay = 0.0001;
        double baseDropout = 0.2;
        try { baseWeightDecay = Double.parseDouble(weightDecayField.getText().trim()); } catch (Exception ignored) {}
        try { baseDropout = Double.parseDouble(dropoutField.getText().trim()); } catch (Exception ignored) {}

        if (learnSpecCoeff == 0) {
            net.setWeightDecay(0);
            net.setDropoutRate(0);
            net.setMiniBatchSize(0);
            net.setLrSchedule("none");
            trainLogArea.append("[学习特化] 纯背诵模式: 无正则化\n");
        } else {
            double scaledWeightDecay = baseWeightDecay * learnSpecCoeff;
            double scaledDropout = Math.min(baseDropout * learnSpecCoeff, 0.8);
            try { net.setMiniBatchSize(Integer.parseInt(batchSizeField.getText().trim())); } catch (Exception ignored) {}
            net.setWeightDecay(scaledWeightDecay);
            net.setDropoutRate(scaledDropout);
            net.setLrSchedule(lrScheduleBox.getSelectedItem().toString());
            trainLogArea.append(String.format("[学习特化] 系数=%.1f | 权重衰减=%.6f | Dropout=%.3f | 调度=%s\n",
                    learnSpecCoeff, scaledWeightDecay, scaledDropout, lrScheduleBox.getSelectedItem()));
        }
    }

    private void toggleMode() {
        dualMode = !dualMode;
        if (dualMode) {
            modeToggleLabel.setText("<html><u>模式: 双流</u></html>");
            trainLogArea.append("[模式切换] 已切换到双流模式\n");
        } else {
            modeToggleLabel.setText("<html><u>模式: 单流</u></html>");
            trainLogArea.append("[模式切换] 已切换到单流模式\n");
        }
    }

    private void updateBiasStatus() {
        if (biasAction == BiasAction.NONE) {
            biasStatusLabel.setText("偏误监测: 关");
            biasStatusLabel.setForeground(Color.GRAY);
        } else if (biasAction == BiasAction.WARN) {
            biasStatusLabel.setText(String.format("偏误监测: 警告 [%.3f,%.3f]", biasLow, biasHigh));
            biasStatusLabel.setForeground(new Color(200, 150, 0));
        } else {
            biasStatusLabel.setText(String.format("偏误监测: 中止 [%.3f,%.3f]", biasLow, biasHigh));
            biasStatusLabel.setForeground(new Color(200, 0, 0));
        }
    }

    private void checkBias(double loss) {
        if (biasAction == BiasAction.NONE) return;
        if (loss >= biasLow && loss <= biasHigh) {
            String msg = String.format("[偏误警告] Loss %.6f 落入不健康范围 [%.3f, %.3f]", loss, biasLow, biasHigh);
            if (biasAction == BiasAction.WARN) {
                SwingUtilities.invokeLater(() -> trainLogArea.append(msg + "\n"));
            } else if (biasAction == BiasAction.ABORT) {
                SwingUtilities.invokeLater(() -> {
                    trainLogArea.append(msg + " -> 中止训练\n");
                    stopTraining();
                });
            }
        }
    }

    private void runAutomation(String folderPath, JComboBox<String> strategyBox,
                               JComboBox<String> methodBox, JCheckBox autoTrainBox) {
        if (folderPath == null || folderPath.trim().isEmpty()) {
            JOptionPane.showMessageDialog(this, "请先选择文件夹");
            return;
        }
        File folder = new File(folderPath);
        if (!folder.isDirectory()) {
            JOptionPane.showMessageDialog(this, "文件夹无效: " + folderPath);
            return;
        }

        File[] files = folder.listFiles((dir, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".txt") || lower.endsWith(".csv") || lower.endsWith(".tsv")
                    || lower.endsWith(".md") || lower.endsWith(".text");
        });

        if (files == null || files.length == 0) {
            JOptionPane.showMessageDialog(this, "文件夹中没有找到文本文件");
            return;
        }

        SampleExtractor.Strategy strategy = SampleExtractor.fromLabel(strategyBox.getSelectedItem().toString());
        TextProcessor.VectorizeMethod method = TextProcessor.VectorizeMethod.valueOf(
                methodBox.getSelectedItem().toString());
        boolean shouldAutoTrain = autoTrainBox.isSelected();

        trainLogArea.append("\n[自动化] 开始批量处理: " + files.length + " 个文件\n");
        trainLogArea.append("[自动化] 策略: " + strategy.label + " | 向量化: " + method + "\n");

        List<SampleExtractor.ExtractedSample> allSamples = new ArrayList<>();
        SampleExtractor extractor = new SampleExtractor(strategy);
        int fileCount = 0;

        for (File file : files) {
            try {
                List<String> lines = java.nio.file.Files.readAllLines(file.toPath(),
                        java.nio.charset.StandardCharsets.UTF_8);
                List<SampleExtractor.ExtractedSample> samples;
                if (strategy == SampleExtractor.Strategy.BILINGUAL_PAIR
                        || strategy == SampleExtractor.Strategy.ARTICLE_TOPIC) {
                    samples = extractor.extractFromLines(lines);
                } else {
                    String fullText = String.join("\n", lines);
                    samples = extractor.extract(Collections.singletonList(fullText));
                }
                allSamples.addAll(samples);
                fileCount++;
                trainLogArea.append("[自动化] " + file.getName() + ": " + samples.size() + " 条样本\n");
            } catch (Exception ex) {
                trainLogArea.append("[自动化] " + file.getName() + " 失败: " + ex.getMessage() + "\n");
            }
        }

        if (allSamples.isEmpty()) {
            trainLogArea.append("[自动化] 未提取到任何样本\n");
            JOptionPane.showMessageDialog(this, "批量处理完成，但未提取到任何样本");
            return;
        }

        originalTexts.clear();
        for (SampleExtractor.ExtractedSample s : allSamples) {
            if (s.inputText != null && !s.inputText.trim().isEmpty()) {
                originalTexts.add(s.inputText.trim());
            }
        }

        try {
            structTextProcessor = new TextProcessor(method, 1000);
            textDataSet = extractor.toDataSet(allSamples, structTextProcessor);

            int outputDim = textDataSet.outputDim();
            textLayerField.setText(structTextProcessor.getVectorSize() + ",32," + outputDim);

            textDataLabel.setText("批量导入: " + allSamples.size() + " 条 | " + fileCount + " 文件 | " + method);
            dataAnalysisArea.setText("=== 批量自动化结果 ===\n处理文件: " + fileCount + " 个\n"
                    + "总计样本: " + allSamples.size() + " 条\n"
                    + "策略: " + strategy.label + "\n"
                    + "向量化: " + method + " | 维度: " + structTextProcessor.getVectorSize()
                    + "\n输出维度: " + outputDim
                    + "\n\n" + textDataSet.getAnalysis());
            dataPreviewArea.setText(textDataSet.preview(20));

            trainLogArea.append("[自动化] 批量处理完成: " + allSamples.size() + " 条样本, "
                    + fileCount + " 个文件\n");

            if (shouldAutoTrain) {
                trainLogArea.append("[自动化] 自动开始训练...\n");
                if (!dualMode) {
                    toggleMode();
                }
                if (dualNetwork == null) {
                    trainLogArea.append("[自动化] 需要先构建双流网络\n");
                    JOptionPane.showMessageDialog(this, "批量处理完成，请先构建双流网络再开始训练");
                    return;
                }
                startDualTraining();
            }
        } catch (Exception ex) {
            trainLogArea.append("[自动化] 数据处理失败: " + ex.getMessage() + "\n");
            JOptionPane.showMessageDialog(this, "批量处理失败: " + ex.getMessage());
        }
    }

    private void startTraining() {
        if (dualMode && dualNetwork != null) {
            startDualTraining();
            return;
        }

        if (enhancedNetwork != null) {
            startEnhancedTraining();
            return;
        }

        if (network == null) {
            JOptionPane.showMessageDialog(this, "请先构建网络");
            return;
        }
        DataSet ds = trainSet != null ? trainSet : dataSet;
        if (ds == null) {
            JOptionPane.showMessageDialog(this, "请先加载数据");
            return;
        }

        int epochs;
        int logInterval;
        try {
            epochs = Integer.parseInt(epochField.getText().trim());
            logInterval = Integer.parseInt(logIntervalField.getText().trim());
            config.setLogInterval(logInterval);
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "参数格式错误");
            return;
        }

        startBtn.setEnabled(false);
        stopBtn.setEnabled(true);
        trainLogArea.setText("");
        trainingChart.clearData();

        applyRegularization(network);

        int earlyStopPatience = 0;
        try { earlyStopPatience = Integer.parseInt(earlyStopField.getText().trim()); } catch (Exception ignored) {}

        trainer = new Trainer(network, ds, testSet);
        if (earlyStopPatience > 0) trainer.setEarlyStopping(earlyStopPatience, 1e-6);

        TrainingCallback uiCallback = new TrainingCallback() {
            @Override
            public void onEpochEnd(int epoch, double loss, double valLoss) {
                checkBias(loss);
                SwingUtilities.invokeLater(() -> {
                    int pct = (int) (100.0 * epoch / epochs);
                    progressBar.setValue(pct);
                    epochLabel.setText("Epoch: " + epoch + "/" + epochs);
                    lossLabel.setText(String.format("Loss: %.6f", loss));
                    trainingChart.addDataPoint(TrainingChart.MetricType.LOSS, loss);
                    trainingChart.addDataPoint(TrainingChart.MetricType.LEARNING_RATE, network.getCurrentLR());
                    if (!Double.isNaN(valLoss)) {
                        trainingChart.addDataPoint(TrainingChart.MetricType.VAL_LOSS, valLoss);
                    }

                    if (testSet != null && epoch % logInterval == 0) {
                        double acc = computeAccuracy();
                        accuracyLabel.setText(String.format("Accuracy: %.2f%%", acc * 100));
                        trainingChart.addDataPoint(TrainingChart.MetricType.ACCURACY, acc);
                    }
                });
            }
        };
        ConsoleTrainingCallback logCallback = new ConsoleTrainingCallback(
                msg -> SwingUtilities.invokeLater(() -> {
                    trainLogArea.append(msg + "\n");
                    trainLogArea.setCaretPosition(trainLogArea.getDocument().getLength());
                }),
                logInterval
        );

        trainer.addCallback(uiCallback);
        trainer.addCallback(logCallback);

        trainThread = new Thread(() -> {
            try {
                lastResult = trainer.train(epochs, logInterval);

                if (testSet != null) {
                    SwingUtilities.invokeLater(() -> {
                        updateConfusionMatrix();
                        double acc = computeAccuracy();
                        accuracyLabel.setText(String.format("Accuracy: %.2f%%", acc * 100));
                        trainLogArea.append(String.format("\n最终准确率: %.2f%%\n", acc * 100));
                    });
                }
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() ->
                        trainLogArea.append("训练出错: " + ex.getMessage() + "\n"));
            } finally {
                SwingUtilities.invokeLater(() -> {
                    startBtn.setEnabled(true);
                    stopBtn.setEnabled(false);
                    progressBar.setValue(100);
                    syncApiServer();
                });
            }
        }, "MaChain-Train");
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private void startEnhancedTraining() {
        if (enhancedNetwork == null) {
            JOptionPane.showMessageDialog(this, "请先构建增强网络");
            return;
        }
        DataSet ds = trainSet != null ? trainSet : dataSet;
        if (ds == null && textDataSet == null) {
            JOptionPane.showMessageDialog(this, "请先加载数据");
            return;
        }
        
        DataSet trainingData = ds != null ? ds : textDataSet;

        int epochs;
        try {
            epochs = Integer.parseInt(epochField.getText().trim());
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "参数格式错误");
            return;
        }

        startBtn.setEnabled(false);
        stopBtn.setEnabled(true);
        trainLogArea.setText("");
        trainingChart.clearData();

        trainLogArea.append("[增强网络] 开始训练 (LSTM + Attention)\n");
        trainLogArea.append("[增强网络] 词汇量: " + enhancedNetwork.getOutputDim() + "\n");

        trainThread = new Thread(() -> {
            try {
                int[][] inputs = new int[trainingData.size()][50];
                double[][] targets = trainingData.getTargets();
                
                if (textProcessor != null && !originalTexts.isEmpty()) {
                    for (int i = 0; i < originalTexts.size() && i < inputs.length; i++) {
                        inputs[i] = textProcessor.textToSequence(originalTexts.get(i), 50);
                    }
                } else {
                    double[][] rawInputs = trainingData.getInputs();
                    for (int i = 0; i < rawInputs.length; i++) {
                        for (int j = 0; j < Math.min(rawInputs[i].length, 50); j++) {
                            inputs[i][j] = (int)(rawInputs[i][j] * 100) % enhancedNetwork.getOutputDim();
                        }
                    }
                }

                for (int epoch = 0; epoch < epochs; epoch++) {
                    final int currentEpoch = epoch;
                    double loss = enhancedNetwork.train(inputs, targets, 1);
                    
                    SwingUtilities.invokeLater(() -> {
                        int pct = (int) (100.0 * currentEpoch / epochs);
                        progressBar.setValue(pct);
                        epochLabel.setText("Epoch: " + currentEpoch + "/" + epochs);
                        lossLabel.setText(String.format("Loss: %.6f", loss));
                        trainingChart.addDataPoint(TrainingChart.MetricType.LOSS, loss);
                        trainLogArea.append(String.format("[Epoch %d] Loss: %.6f\n", currentEpoch, loss));
                    });
                }

                SwingUtilities.invokeLater(() -> {
                    trainLogArea.append("[增强网络] 训练完成\n");
                    syncApiServer();
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() ->
                        trainLogArea.append("训练出错: " + ex.getMessage() + "\n"));
            } finally {
                SwingUtilities.invokeLater(() -> {
                    startBtn.setEnabled(true);
                    stopBtn.setEnabled(false);
                    progressBar.setValue(100);
                });
            }
        }, "MaChain-Enhanced-Train");
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private double computeAccuracy() {
        if (testSet == null || network == null) return 0;
        double[][] inputs = testSet.getInputs();
        double[][] targets = testSet.getTargets();
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] output = network.forward(inputs[i]);
            int predClass = outputMode.predictClass(output);
            int trueClass = 0;
            for (int j = 1; j < targets[i].length; j++) {
                if (targets[i][j] > targets[i][trueClass]) trueClass = j;
            }
            if (predClass == trueClass) correct++;
        }
        return (double) correct / inputs.length;
    }

    private void updateConfusionMatrix() {
        if (testSet == null || network == null) return;
        double[][] inputs = testSet.getInputs();
        double[][] targets = testSet.getTargets();
        double[][] predictions = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = network.forward(inputs[i]);
        }
        int[][] matrix = ConfusionMatrixPanel.computeConfusionMatrix(predictions, targets);
        confusionMatrixPanel.setMatrix(matrix, testSet.getLabelNames());
    }

    private void startDualTraining() {
        DataSet sTrain = structTrainSet != null ? structTrainSet : structDataSet;
        DataSet tTrain = textTrainSet != null ? textTrainSet : textDataSet;

        if (sTrain == null || tTrain == null) {
            JOptionPane.showMessageDialog(this, "双流模式需要同时加载结构化和文本数据");
            return;
        }

        int epochs;
        int logInterval;
        try {
            epochs = Integer.parseInt(epochField.getText().trim());
            logInterval = Integer.parseInt(logIntervalField.getText().trim());
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "参数格式错误");
            return;
        }

        startBtn.setEnabled(false);
        stopBtn.setEnabled(true);
        trainLogArea.setText("");
        trainingChart.clearData();
        trainLogArea.append("=== 双流训练模式 ===\n");
        trainLogArea.append("结构化数据: " + sTrain.size() + " 条\n");
        trainLogArea.append("文本数据: " + tTrain.size() + " 条\n");
        trainLogArea.append("桥接维度: " + dualNetwork.getBridge().getBridgeDim() + "\n");
        trainLogArea.append("交叉蒸馏权重: " + dualNetwork.getCrossLossWeight() + "\n\n");

        applyRegularization(dualNetwork.getStructNetwork());
        applyRegularization(dualNetwork.getTextNetwork());

        int earlyStopPatience = 0;
        try { earlyStopPatience = Integer.parseInt(earlyStopField.getText().trim()); } catch (Exception ignored) {}

        dualTrainer = new DualTrainer(dualNetwork, sTrain, structTestSet, tTrain, textTestSet);
        if (earlyStopPatience > 0) dualTrainer.setEarlyStopping(earlyStopPatience, 1e-6);

        final int totalEpochs = epochs;
        TrainingCallback uiCallback = new TrainingCallback() {
            @Override
            public void onEpochEnd(int epoch, double loss, double valLoss) {
                checkBias(loss);
                SwingUtilities.invokeLater(() -> {
                    int pct = (int) (100.0 * epoch / totalEpochs);
                    progressBar.setValue(pct);
                    epochLabel.setText("Epoch: " + epoch + "/" + totalEpochs);
                    lossLabel.setText(String.format("Total: %.6f", loss));
                    crossLossLabel.setText(String.format("Cross: %.6f", dualNetwork.getLastCrossLoss()));
                    trainingChart.addDataPoint(TrainingChart.MetricType.LOSS, loss);
                    trainingChart.addDataPoint(TrainingChart.MetricType.LEARNING_RATE,
                            dualNetwork.getStructNetwork().getCurrentLR());
                    if (!Double.isNaN(valLoss)) {
                        trainingChart.addDataPoint(TrainingChart.MetricType.VAL_LOSS, valLoss);
                    }
                });
            }
        };
        ConsoleTrainingCallback logCallback = new ConsoleTrainingCallback(
                msg -> SwingUtilities.invokeLater(() -> {
                    trainLogArea.append(msg + "\n");
                    trainLogArea.setCaretPosition(trainLogArea.getDocument().getLength());
                }),
                logInterval
        );

        dualTrainer.addCallback(uiCallback);
        dualTrainer.addCallback(logCallback);

        trainThread = new Thread(() -> {
            try {
                lastResult = dualTrainer.train(epochs, logInterval);
                SwingUtilities.invokeLater(() -> {
                    trainLogArea.append("\n=== 双流训练完成 ===\n");
                    trainLogArea.append(String.format("结构化Loss: %.6f\n", dualNetwork.getLastStructLoss()));
                    trainLogArea.append(String.format("文本Loss: %.6f\n", dualNetwork.getLastTextLoss()));
                    trainLogArea.append(String.format("交叉蒸馏Loss: %.6f\n", dualNetwork.getLastCrossLoss()));
                    trainLogArea.append(String.format("隶属度: [%.3f, %.3f]\n",
                            dualNetwork.getBridge().getMembershipWeights()[0],
                            dualNetwork.getBridge().getMembershipWeights()[1]));
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() ->
                        trainLogArea.append("训练出错: " + ex.getMessage() + "\n"));
            } finally {
                SwingUtilities.invokeLater(() -> {
                    startBtn.setEnabled(true);
                    stopBtn.setEnabled(false);
                    progressBar.setValue(100);
                });
            }
        }, "MaChain-DualTrain");
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private void stopTraining() {
        if (dualMode && dualTrainer != null) {
            dualTrainer.stop();
            trainLogArea.append("用户请求停止...\n");
        } else if (trainer != null) {
            trainer.stop();
            trainLogArea.append("用户请求停止...\n");
        }
    }

    // ========== 可视化面板 ==========
    private JPanel buildVisualizationPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton updateBtn = new JButton("更新混淆矩阵");
        updateBtn.addActionListener(e -> {
            if (testSet != null && network != null) {
                updateConfusionMatrix();
                JOptionPane.showMessageDialog(this, "混淆矩阵已更新");
            } else {
                JOptionPane.showMessageDialog(this, "请先完成训练并划分测试集");
            }
        });
        topPanel.add(updateBtn);
        panel.add(topPanel, BorderLayout.NORTH);

        confusionMatrixPanel = new ConfusionMatrixPanel();
        JScrollPane sp = new JScrollPane(confusionMatrixPanel);
        sp.setBorder(BorderFactory.createTitledBorder("混淆矩阵 (分类任务)"));
        panel.add(sp, BorderLayout.CENTER);

        return panel;
    }

    // ========== 输出面板 ==========
    private JPanel buildOutputPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        topPanel.add(new JLabel("输出格式:"));
        outputFormatBox = new JComboBox<>(new String[]{"csv", "json"});
        topPanel.add(outputFormatBox);

        JButton exportResultBtn = new JButton("导出训练结果");
        exportResultBtn.addActionListener(e -> exportTrainingResult());
        topPanel.add(exportResultBtn);

        JButton exportPredBtn = new JButton("导出预测结果");
        exportPredBtn.addActionListener(e -> exportPredictions());
        topPanel.add(exportPredBtn);

        JButton exportModelBtn = new JButton("导出模型信息");
        exportModelBtn.addActionListener(e -> exportModelInfo());
        topPanel.add(exportModelBtn);

        JButton openDirBtn = new JButton("打开输出目录");
        openDirBtn.addActionListener(e -> openOutputDir());
        topPanel.add(openDirBtn);

        panel.add(topPanel, BorderLayout.NORTH);

        outputLogArea = new JTextArea();
        outputLogArea.setEditable(false);
        outputLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        outputLogArea.setText("输出模块说明:\n" +
                "  - 导出训练结果: 将loss曲线数据导出为CSV/JSON\n" +
                "  - 导出预测结果: 将模型对数据集的预测导出\n" +
                "  - 导出模型信息: 将网络结构和参数导出\n" +
                "  - 输出目录: " + outputManager.getOutputDir() + "\n\n" +
                "支持的数据类型:\n" +
                "  - 数值数据 (CSV格式)\n" +
                "  - 文本数据 (自动向量化)\n\n" +
                "输出模式:\n" +
                "  - 回归: 连续值\n" +
                "  - 分类: 概率分布\n" +
                "  - 多标签: 独立概率\n");
        JScrollPane sp = new JScrollPane(outputLogArea);
        sp.setBorder(BorderFactory.createTitledBorder("输出日志"));
        panel.add(sp, BorderLayout.CENTER);

        return panel;
    }

    private void exportTrainingResult() {
        if (lastResult == null) {
            JOptionPane.showMessageDialog(this, "没有可导出的训练结果, 请先完成训练");
            return;
        }
        try {
            String format = outputFormatBox.getSelectedItem().toString();
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String path = outputManager.exportTrainingResult(lastResult, format, "training_" + timestamp);
            outputLogArea.append("训练结果已导出: " + path + "\n");
        } catch (Exception ex) {
            outputLogArea.append("导出失败: " + ex.getMessage() + "\n");
        }
    }

    private void exportPredictions() {
        if (network == null) {
            JOptionPane.showMessageDialog(this, "请先构建或加载网络");
            return;
        }
        DataSet ds = dataSet;
        if (ds == null) {
            JOptionPane.showMessageDialog(this, "请先加载数据");
            return;
        }
        try {
            String format = outputFormatBox.getSelectedItem().toString();
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String path = outputManager.exportPredictions(network, ds, format, "predictions_" + timestamp);
            outputLogArea.append("预测结果已导出: " + path + "\n");
        } catch (Exception ex) {
            outputLogArea.append("导出失败: " + ex.getMessage() + "\n");
        }
    }

    private void exportModelInfo() {
        if (network == null) {
            JOptionPane.showMessageDialog(this, "请先构建或加载网络");
            return;
        }
        try {
            String format = outputFormatBox.getSelectedItem().toString();
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String path = outputManager.exportModelInfo(network, format, "model_" + timestamp);
            outputLogArea.append("模型信息已导出: " + path + "\n");
        } catch (Exception ex) {
            outputLogArea.append("导出失败: " + ex.getMessage() + "\n");
        }
    }

    private void openOutputDir() {
        File dir = new File(outputManager.getOutputDir());
        dir.mkdirs();
        try {
            Desktop.getDesktop().open(dir);
        } catch (Exception ex) {
            outputLogArea.append("无法打开目录: " + ex.getMessage() + "\n");
        }
    }

    // ========== 存档操作 (按钮在底部右侧) ==========
    private void saveProgress() {
        Network target = dualMode ? dualNetwork.getStructNetwork() : network;
        if (target == null) {
            JOptionPane.showMessageDialog(this, "没有可保存的网络");
            return;
        }
        String note = JOptionPane.showInputDialog(this, "备注(可选):", "保存进度");
        if (note == null) return;
        try {
            String path = saveManager.saveProgress(target, dataSet, note);
            JOptionPane.showMessageDialog(this, "进度已保存:\n" + path);
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "保存失败: " + ex.getMessage());
        }
    }

    private void saveSkill() {
        Network target = dualMode ? dualNetwork.getStructNetwork() : network;
        if (target == null) {
            JOptionPane.showMessageDialog(this, "没有可保存的网络");
            return;
        }
        String name = JOptionPane.showInputDialog(this, "Skill名称:", "保存Skill");
        if (name == null || name.trim().isEmpty()) return;
        try {
            String path = saveManager.saveSkill(target, dataSet, name.trim());
            JOptionPane.showMessageDialog(this, "Skill已保存:\n" + path);
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "保存失败: " + ex.getMessage());
        }
    }

    private void loadSaveDialog() {
        JFileChooser fc = new JFileChooser(saveManager.getSaveDir());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("MaChain存档", "mcsave", "mcskill"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            SaveManager.SaveData sd = saveManager.load(fc.getSelectedFile().getAbsolutePath());
            network = sd.network;
            trainLogArea.setText("");
            trainLogArea.append("已加载存档: " + fc.getSelectedFile().getName() + "\n");
            trainLogArea.append("类型: " + sd.type + "\n");
            trainLogArea.append("备注: " + sd.note + "\n");
            trainLogArea.append("网络: " + network.getArchitecture() + "\n");
            trainLogArea.append("参数量: " + network.countParameters() + "\n");
            trainLogArea.append("已训练轮数: " + network.getTotalEpochs() + "\n");
            trainLogArea.append("最优Loss: " + network.getBestLoss() + "\n\n");
            trainLogArea.append("--- 数据信息 ---\n" + sd.dataInfo + "\n");

            layerField.setText(arrayToString(network.getLayerSizes()));
            lrField.setText(String.valueOf(network.getLearningRate()));
            momentumField.setText(String.valueOf(network.getMomentum()));

            JOptionPane.showMessageDialog(this, "存档已加载");
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载失败: " + ex.getMessage());
        }
    }

    private void saveVocabulary() {
        TextProcessor tp = textProcessor != null ? textProcessor : structTextProcessor;
        if (tp == null || !tp.isBuilt()) {
            JOptionPane.showMessageDialog(this, "没有可用的词库，请先训练或加载数据");
            return;
        }
        JFileChooser fc = new JFileChooser(saveManager.getSaveDir());
        fc.setSelectedFile(new java.io.File("vocabulary_" + new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date()) + ".mcvocab"));
        if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            String path = fc.getSelectedFile().getAbsolutePath();
            if (!path.endsWith(".mcvocab")) path += ".mcvocab";
            tp.saveVocabulary(path);
            trainLogArea.append("[词库] 已保存: " + path + " | 词汇量: " + tp.getVectorSize() + "\n");
            JOptionPane.showMessageDialog(this, "词库已保存:\n" + path + "\n词汇量: " + tp.getVectorSize());
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "保存词库失败: " + ex.getMessage());
        }
    }

    private void loadVocabulary() {
        JFileChooser fc = new JFileChooser(saveManager.getSaveDir());
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("MaChain词库", "mcvocab"));
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            String path = fc.getSelectedFile().getAbsolutePath();
            TextProcessor tp = new TextProcessor(TextProcessor.VectorizeMethod.TF_IDF, 1000);
            tp.loadVocabulary(path);
            textProcessor = tp;
            structTextProcessor = tp;
            trainLogArea.append("[词库] 已加载: " + fc.getSelectedFile().getName() + " | 词汇量: " + tp.getVectorSize() + "\n");
            JOptionPane.showMessageDialog(this, "词库已加载:\n" + fc.getSelectedFile().getName() + "\n词汇量: " + tp.getVectorSize());
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "加载词库失败: " + ex.getMessage());
        }
    }

    private String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        return sb.toString();
    }

    private int[] parseIntArray(String s) {
        String[] parts = s.trim().split(",");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }
        return result;
    }

    private void toggleApiServer() {
        if (apiServer != null && apiServer.isRunning()) {
            apiServer.stop();
            apiServer = null;
            apiToggleBtn.setText("启动API");
            apiToggleBtn.setForeground(new Color(0, 100, 0));
            apiStatusLabel.setText("已停止");
            apiStatusLabel.setForeground(Color.GRAY);
            apiPortField.setEnabled(true);
            appHistoryArea.append("[API] 服务已停止\n");
            return;
        }

        int port;
        try {
            port = Integer.parseInt(apiPortField.getText().trim());
        } catch (NumberFormatException e) {
            apiStatusLabel.setText("端口无效");
            return;
        }

        apiServer = new ApiServer(port);

        Network net = appNetwork != null ? appNetwork : network;
        TextProcessor tp = appTextProcessor != null ? appTextProcessor : textProcessor;
        apiServer.setModel(net, tp, appOutputMode);
        apiServer.setTextGenerator(textGenerator);
        apiServer.setEnhancedNetwork(enhancedNetwork);

        try {
            apiServer.start();
            apiToggleBtn.setText("停止API");
            apiToggleBtn.setForeground(new Color(180, 0, 0));
            apiStatusLabel.setText("运行中 ://" + "localhost:" + port);
            apiStatusLabel.setForeground(new Color(0, 128, 0));
            apiPortField.setEnabled(false);
            appHistoryArea.append("[API] 服务已启动 → http://localhost:" + port + "/\n");
            appHistoryArea.append("[API]   POST /api/predict  — 数值/文本推理\n");
            appHistoryArea.append("[API]   POST /api/generate — 自主文本生成\n");
            appHistoryArea.append("[API]   GET  /api/status   — 模型状态\n");
        } catch (Exception e) {
            apiStatusLabel.setText("启动失败: " + e.getMessage());
            apiStatusLabel.setForeground(Color.RED);
            appHistoryArea.append("[API] 启动失败: " + e.getMessage() + "\n");
        }
    }
}

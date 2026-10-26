package machain.io;

import machain.network.Network;
import machain.network.DualStreamNetwork;
import machain.data.DataSet;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class SaveManager {

    private final String saveDir;
    private final String skillDir;

    public SaveManager(String baseDir) {
        this.saveDir = baseDir + File.separator + "saves";
        this.skillDir = saveDir + File.separator + "skills";
        new File(saveDir).mkdirs();
        new File(skillDir).mkdirs();
    }

    public String saveProgress(Network net, DataSet data, String note) throws IOException {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = "progress_" + timestamp + ".mcsave";
        File file = new File(saveDir, filename);

        SaveData sd = new SaveData();
        sd.network = net;
        sd.dataInfo = data != null ? data.getAnalysis() : "无数据";
        sd.note = note;
        sd.timestamp = System.currentTimeMillis();
        sd.type = SaveType.PROGRESS;

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(sd);
        }
        return file.getAbsolutePath();
    }

    public String saveSkill(Network net, DataSet data, String skillName) throws IOException {
        String filename = skillName.replaceAll("[^a-zA-Z0-9_\\-]", "_") + ".mcskill";
        File file = new File(skillDir, filename);

        SaveData sd = new SaveData();
        sd.network = net;
        sd.dataInfo = data != null ? data.getAnalysis() : "无数据";
        sd.note = skillName;
        sd.timestamp = System.currentTimeMillis();
        sd.type = SaveType.SKILL;

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(sd);
        }
        return file.getAbsolutePath();
    }

    public SaveData load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (SaveData) ois.readObject();
        }
    }

    public String[][] listSaves() {
        List<String[]> saves = new ArrayList<>();
        File dir = new File(saveDir);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".mcsave") || name.endsWith(".mcskill"));
        if (files != null) {
            for (File f : files) {
                String type = f.getName().endsWith(".mcskill") ? "SKILL" : "PROGRESS";
                saves.add(new String[]{f.getName(), type, formatSize(f.length()),
                        new SimpleDateFormat("yyyy-MM-dd HH:mm").format(new Date(f.lastModified()))});
            }
        }
        saves.sort((a, b) -> b[0].compareTo(a[0]));
        return saves.toArray(new String[0][]);
    }

    public boolean deleteSave(String filename) {
        File f = new File(saveDir, filename);
        if (!f.exists()) f = new File(skillDir, filename);
        return f.exists() && f.delete();
    }

    private String formatSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        return String.format("%.1f MB", bytes / (1024.0 * 1024));
    }

    public String saveDualProgress(DualStreamNetwork dualNet, DataSet structData, DataSet textData, String note) throws IOException {
        String timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date());
        String filename = "dual_progress_" + timestamp + ".mcsave";
        File file = new File(saveDir, filename);

        SaveData sd = new SaveData();
        sd.dualNetwork = dualNet;
        sd.dataInfo = (structData != null ? structData.getAnalysis() : "无结构化数据") + "\n---\n" +
                      (textData != null ? textData.getAnalysis() : "无文本数据");
        sd.note = note;
        sd.timestamp = System.currentTimeMillis();
        sd.type = SaveType.PROGRESS;

        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(file))) {
            oos.writeObject(sd);
        }
        return file.getAbsolutePath();
    }

    public String saveDualSkill(DualStreamNetwork dualNet, DataSet structData, DataSet textData, String skillName) throws IOException {
        String filename = skillName.replaceAll("[^a-zA-Z0-9_\\-]", "_") + "_dual.mcskill";
        File file = new File(skillDir, filename);

        SaveData sd = new SaveData();
        sd.dualNetwork = dualNet;
        sd.dataInfo = (structData != null ? structData.getAnalysis() : "无结构化数据") + "\n---\n" +
                      (textData != null ? textData.getAnalysis() : "无文本数据");
        sd.note = skillName;
        sd.timestamp = System.currentTimeMillis();
        sd.type = SaveType.SKILL;

        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(file))) {
            oos.writeObject(sd);
        }
        return file.getAbsolutePath();
    }

    public String getSaveDir() { return saveDir; }
    public String getSkillDir() { return skillDir; }

    public enum SaveType { PROGRESS, SKILL }

    public static class SaveData implements Serializable {
        private static final long serialVersionUID = 2L;
        public Network network;
        public DualStreamNetwork dualNetwork;
        public String dataInfo;
        public String note;
        public long timestamp;
        public SaveType type;

        public boolean isDualStream() {
            return dualNetwork != null;
        }
    }
}

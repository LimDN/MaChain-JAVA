package machain.gpu;

public class GPUAccelerator {

    private static GPUAccelerator instance;
    private GPUMatrixOps matrixOps;
    private boolean enabled = false;
    private boolean lwjglPresent = false;

    private GPUAccelerator() {
        try {
            Class.forName("org.lwjgl.opencl.CL");
            lwjglPresent = true;
            this.matrixOps = new GPUMatrixOps();
        } catch (Throwable t) {
            lwjglPresent = false;
            System.out.println("[GPU] LWJGL未加载，GPU加速不可用");
        }
    }

    public static synchronized GPUAccelerator getInstance() {
        if (instance == null) {
            instance = new GPUAccelerator();
        }
        return instance;
    }

    public boolean isEnabled() {
        return lwjglPresent && enabled && matrixOps != null && matrixOps.isGpuAvailable();
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public GPUMatrixOps getMatrixOps() {
        return matrixOps;
    }

    public String getStatus() {
        if (!lwjglPresent || matrixOps == null) {
            return "LWJGL未安装 (CPU模式)";
        }
        if (!matrixOps.isGpuAvailable()) {
            return "GPU不可用 (CPU模式)";
        }
        CUDAContext ctx = CUDAContext.getInstance();
        return String.format("GPU: %s (%d MB) - %s",
                ctx.getDeviceName(),
                ctx.getTotalMemory() / 1024 / 1024,
                enabled ? "已启用" : "已禁用");
    }

    public void initialize() {
        if (!lwjglPresent) {
            System.out.println("GPU加速器: LWJGL未加载，使用CPU模式");
            return;
        }
        try {
            CUDAContext ctx = CUDAContext.getInstance();
            if (ctx.initialize()) {
                this.matrixOps = new GPUMatrixOps();
                this.enabled = matrixOps.isGpuAvailable();
                System.out.println("GPU加速器已就绪");
            } else {
                System.out.println("GPU加速器不可用，使用CPU模式");
            }
        } catch (Throwable t) {
            System.err.println("[GPU] 初始化失败: " + t.getMessage());
            enabled = false;
        }
    }

    public void shutdown() {
        if (!lwjglPresent) return;
        try {
            CUDAContext ctx = CUDAContext.getInstance();
            ctx.shutdown();
        } catch (Throwable ignored) {}
    }
}

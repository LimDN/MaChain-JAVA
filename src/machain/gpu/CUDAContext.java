package machain.gpu;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class CUDAContext {

    private static CUDAContext instance;
    private boolean initialized = false;
    private boolean openclAvailable = false;

    private long platformId;
    private long deviceId;
    private long context;
    private long commandQueue;

    private int deviceCount = 0;
    private String deviceName = "N/A";
    private String platformName = "N/A";
    private long totalMemory = 0;
    private int maxComputeUnits = 0;

    private CUDAContext() {}

    public static synchronized CUDAContext getInstance() {
        if (instance == null) {
            instance = new CUDAContext();
        }
        return instance;
    }

    public boolean initialize() {
        if (initialized) return openclAvailable;

        try {
            System.out.println("[GPU] 正在初始化 OpenCL...");
            try {
                CL.create();
            } catch (IllegalStateException e) {
                System.out.println("[GPU] OpenCL 已处于初始化状态，跳过重复创建");
            }

            try (MemoryStack stack = stackPush()) {
                IntBuffer errBuf = stack.mallocInt(1);

                PointerBuffer platformIds = stack.mallocPointer(8);
                int[] numPlatforms = new int[1];
                int clErr = clGetPlatformIDs(platformIds, numPlatforms);
                if (clErr != CL_SUCCESS || numPlatforms[0] == 0) {
                    System.err.println("[GPU] 未找到 OpenCL 平台 (err=" + clErr + ")");
                    finishInit(false);
                    return false;
                }

                platformId = platformIds.get(0);
                platformName = queryPlatformString(platformId, CL_PLATFORM_NAME);
                String platformVendor = queryPlatformString(platformId, CL_PLATFORM_VENDOR);
                System.out.println("[GPU] 平台: " + platformName + " (" + platformVendor + ")");

                CLCapabilities platformCaps = CL.createPlatformCapabilities(platformId);
                if (platformCaps == null || !platformCaps.OpenCL10) {
                    System.err.println("[GPU] 平台不支持 OpenCL 1.0");
                    finishInit(false);
                    return false;
                }

                PointerBuffer deviceIds = stack.mallocPointer(8);
                int[] numDevices = new int[1];
                clErr = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, deviceIds, numDevices);
                if (clErr != CL_SUCCESS || numDevices[0] == 0) {
                    clErr = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceIds, numDevices);
                    if (clErr != CL_SUCCESS || numDevices[0] == 0) {
                        System.err.println("[GPU] 未找到 OpenCL 设备");
                        finishInit(false);
                        return false;
                    }
                }

                deviceId = deviceIds.get(0);
                deviceCount = numDevices[0];
                deviceName = queryDeviceString(deviceId, CL_DEVICE_NAME);
                totalMemory = queryDeviceLong(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
                maxComputeUnits = (int) queryDeviceLong(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);

                System.out.println("[GPU] 设备: " + deviceName);
                System.out.println("[GPU] 显存: " + (totalMemory / 1024 / 1024) + " MB, 计算单元: " + maxComputeUnits);

                CLCapabilities deviceCaps = CL.createDeviceCapabilities(deviceId, platformCaps);
                if (deviceCaps == null) {
                    System.err.println("[GPU] 无法创建设备能力");
                    finishInit(false);
                    return false;
                }

                PointerBuffer ctxProps = stack.mallocPointer(3);
                ctxProps.put(CL_CONTEXT_PLATFORM).put(platformId).put(0).flip();

                context = clCreateContext(ctxProps, deviceId, null, 0, errBuf);
                if (errBuf.get(0) != CL_SUCCESS) {
                    System.err.println("[GPU] 创建上下文失败 (err=" + errBuf.get(0) + ")");
                    finishInit(false);
                    return false;
                }

                commandQueue = clCreateCommandQueue(context, deviceId, 0, errBuf);
                if (errBuf.get(0) != CL_SUCCESS) {
                    System.err.println("[GPU] 创建命令队列失败 (err=" + errBuf.get(0) + ")");
                    clReleaseContext(context);
                    context = 0;
                    finishInit(false);
                    return false;
                }

                openclAvailable = true;
                initialized = true;
                System.out.println("[GPU] OpenCL 初始化成功 ✓");
                return true;

            }
        } catch (Throwable t) {
            System.err.println("[GPU] OpenCL 初始化失败: " + t.getClass().getSimpleName() + " - " + t.getMessage());
            finishInit(false);
            return false;
        }
    }

    private void finishInit(boolean success) {
        openclAvailable = success;
        initialized = true;
    }

    private String queryPlatformString(long platform, int param) {
        try (MemoryStack stack = stackPush()) {
            PointerBuffer sizeBuf = stack.mallocPointer(1);
            clGetPlatformInfo(platform, param, (ByteBuffer) null, sizeBuf);
            int size = (int) sizeBuf.get(0);
            ByteBuffer buf = stack.malloc(size);
            clGetPlatformInfo(platform, param, buf, null);
            return memUTF8(buf, size - 1);
        } catch (Exception e) { return "Unknown"; }
    }

    private String queryDeviceString(long device, int param) {
        try (MemoryStack stack = stackPush()) {
            PointerBuffer sizeBuf = stack.mallocPointer(1);
            clGetDeviceInfo(device, param, (ByteBuffer) null, sizeBuf);
            int size = (int) sizeBuf.get(0);
            ByteBuffer buf = stack.malloc(size);
            clGetDeviceInfo(device, param, buf, null);
            return memUTF8(buf, size - 1);
        } catch (Exception e) { return "Unknown"; }
    }

    private long queryDeviceLong(long device, int param) {
        try (MemoryStack stack = stackPush()) {
            LongBuffer buf = stack.mallocLong(1);
            PointerBuffer sizeRet = stack.mallocPointer(1);
            clGetDeviceInfo(device, param, buf, sizeRet);
            return buf.get(0);
        } catch (Exception e) { return 0; }
    }

    public boolean isInitialized() { return initialized; }
    public boolean isLwjglAvailable() { return openclAvailable; }
    public int getDeviceCount() { return deviceCount; }
    public String getDeviceName() { return deviceName; }
    public long getTotalMemory() { return totalMemory; }
    public int getCurrentDevice() { return 0; }
    public long getContextHandle() { return context; }
    public long getDeviceHandle() { return deviceId; }
    public long getCommandQueue() { return commandQueue; }
    public String getPlatformName() { return platformName; }
    public int getMaxComputeUnits() { return maxComputeUnits; }

    public void shutdown() {
        if (commandQueue != 0) { clReleaseCommandQueue(commandQueue); commandQueue = 0; }
        if (context != 0) { clReleaseContext(context); context = 0; }
        initialized = false;
        openclAvailable = false;
        try { CL.destroy(); } catch (Exception ignored) {}
        System.out.println("[GPU] OpenCL 已关闭");
    }
}

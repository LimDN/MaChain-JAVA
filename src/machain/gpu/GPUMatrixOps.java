package machain.gpu;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class GPUMatrixOps {

    private static final String MATMUL_KERNEL =
        "__kernel void matmul(\n" +
        "    __global const float *A,\n" +
        "    __global const float *B,\n" +
        "    __global float *C,\n" +
        "    const int M, const int N, const int K)\n" +
        "{\n" +
        "    int row = get_global_id(0);\n" +
        "    int col = get_global_id(1);\n" +
        "    if (row >= M || col >= N) return;\n" +
        "    float sum = 0.0f;\n" +
        "    for (int p = 0; p < K; p++) {\n" +
        "        sum += A[row * K + p] * B[p * N + col];\n" +
        "    }\n" +
        "    C[row * N + col] = sum;\n" +
        "}\n";

    private static final String MATVEC_KERNEL =
        "__kernel void matvec(\n" +
        "    __global const float *A,\n" +
        "    __global const float *x,\n" +
        "    __global const float *bias,\n" +
        "    __global float *y,\n" +
        "    const int M, const int N)\n" +
        "{\n" +
        "    int row = get_global_id(0);\n" +
        "    if (row >= M) return;\n" +
        "    float sum = bias[row];\n" +
        "    for (int j = 0; j < N; j++) {\n" +
        "        sum += A[row * N + j] * x[j];\n" +
        "    }\n" +
        "    y[row] = sum;\n" +
        "}\n";

    private static final String ACTIVATE_KERNEL =
        "__kernel void activate(\n" +
        "    __global float *data,\n" +
        "    const int n, const int type)\n" +
        "{\n" +
        "    int i = get_global_id(0);\n" +
        "    if (i >= n) return;\n" +
        "    float v = data[i];\n" +
        "    if (type == 0) data[i] = 1.0f / (1.0f + exp(-v));\n" +
        "    else if (type == 1) data[i] = v > 0.0f ? v : 0.0f;\n" +
        "    else if (type == 2) data[i] = tanh(v);\n" +
        "}\n";

    private final boolean gpuAvailable;
    private final CUDAContext ctx;

    private long matmulProgram = 0;
    private long matmulKernel = 0;
    private long matvecProgram = 0;
    private long matvecKernel = 0;
    private long activateProgram = 0;
    private long activateKernel = 0;

    private long gpuBufA = 0;
    private long gpuBufB = 0;
    private long gpuBufC = 0;
    private long gpuBufX = 0;
    private long gpuBufBias = 0;
    private long gpuBufY = 0;

    private long capacityA = 0;
    private long capacityB = 0;
    private long capacityC = 0;
    private long capacityX = 0;
    private long capacityBias = 0;
    private long capacityY = 0;

    private int matmulCount = 0;
    private int matvecCount = 0;
    private long totalGpuTimeNanos = 0;

    public GPUMatrixOps() {
        this.ctx = CUDAContext.getInstance();
        this.gpuAvailable = ctx.isInitialized() && ctx.isLwjglAvailable();

        if (gpuAvailable) {
            compileKernels();
        }
    }

    private void compileKernels() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer errBuf = stack.mallocInt(1);
            long cq = ctx.getCommandQueue();
            long cx = ctx.getContextHandle();

            matmulProgram = compileProgram(cx, cq, MATMUL_KERNEL, errBuf);
            if (matmulProgram != 0) {
                matmulKernel = clCreateKernel(matmulProgram, stackUTF8("matmul"), errBuf);
                if (errBuf.get(0) != CL_SUCCESS) matmulKernel = 0;
            }

            matvecProgram = compileProgram(cx, cq, MATVEC_KERNEL, errBuf);
            if (matvecProgram != 0) {
                matvecKernel = clCreateKernel(matvecProgram, stackUTF8("matvec"), errBuf);
                if (errBuf.get(0) != CL_SUCCESS) matvecKernel = 0;
            }

            activateProgram = compileProgram(cx, cq, ACTIVATE_KERNEL, errBuf);
            if (activateProgram != 0) {
                activateKernel = clCreateKernel(activateProgram, stackUTF8("activate"), errBuf);
                if (errBuf.get(0) != CL_SUCCESS) activateKernel = 0;
            }

            System.out.println("[GPU] Kernel编译完成: matmul=" + (matmulKernel != 0) +
                    " matvec=" + (matvecKernel != 0) + " activate=" + (activateKernel != 0));
        } catch (Exception e) {
            System.err.println("[GPU] Kernel编译异常: " + e.getMessage());
        }
    }

    private long compileProgram(long context, long queue, String source, IntBuffer errBuf) {
        try (MemoryStack stack = stackPush()) {
            long program = clCreateProgramWithSource(context, source, errBuf);
            if (errBuf.get(0) != CL_SUCCESS) return 0;

            PointerBuffer devices = stack.mallocPointer(1);
            devices.put(0, ctx.getDeviceHandle());

            int buildErr = clBuildProgram(program, devices, "", null, 0);
            if (buildErr != CL_SUCCESS) {
                try {
                    PointerBuffer logSize = stack.mallocPointer(1);
                    clGetProgramBuildInfo(program, ctx.getDeviceHandle(), CL_PROGRAM_BUILD_LOG,
                            (ByteBuffer) null, logSize);
                    ByteBuffer logBuf = stack.malloc((int) logSize.get(0));
                    clGetProgramBuildInfo(program, ctx.getDeviceHandle(), CL_PROGRAM_BUILD_LOG,
                            logBuf, null);
                    System.err.println("[GPU] 编译日志: " + memUTF8(logBuf));
                } catch (Exception ignored) {}
                clReleaseProgram(program);
                return 0;
            }
            return program;
        }
    }

    public boolean isGpuAvailable() {
        return gpuAvailable && matmulKernel != 0 && matvecKernel != 0;
    }

    public double[] matvecGPU(double[][] Ad, double[] xd, double[] biasd) {
        if (!isGpuAvailable()) return matvecCPU(Ad, xd, biasd);

        long t0 = System.nanoTime();
        int M = Ad.length;
        int N = Ad[0].length;

        try (MemoryStack stack = stackPush()) {
            IntBuffer errBuf = stack.mallocInt(1);
            long cq = ctx.getCommandQueue();

            long bufA = ensureBuffer(gpuBufA, capacityA, (long) M * N * 4, CL_MEM_READ_ONLY, errBuf);
            gpuBufA = bufA; capacityA = (long) M * N * 4;

            long bufX = ensureBuffer(gpuBufX, capacityX, (long) N * 4, CL_MEM_READ_ONLY, errBuf);
            gpuBufX = bufX; capacityX = (long) N * 4;

            long bufBias = ensureBuffer(gpuBufBias, capacityBias, (long) M * 4, CL_MEM_READ_ONLY, errBuf);
            gpuBufBias = bufBias; capacityBias = (long) M * 4;

            long bufY = ensureBuffer(gpuBufY, capacityY, (long) M * 4, CL_MEM_WRITE_ONLY, errBuf);
            gpuBufY = bufY; capacityY = (long) M * 4;

            // 使用堆分配避免栈溢出（大矩阵时MemoryStack可能不够）
            FloatBuffer hostA = BufferUtils.createFloatBuffer(M * N);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    hostA.put(i * N + j, (float) Ad[i][j]);
            clEnqueueWriteBuffer(cq, bufA, true, 0, hostA.rewind(), null, null);

            FloatBuffer hostX = BufferUtils.createFloatBuffer(N);
            for (int i = 0; i < N; i++) hostX.put(i, (float) xd[i]);
            clEnqueueWriteBuffer(cq, bufX, true, 0, hostX.rewind(), null, null);

            FloatBuffer hostBias = BufferUtils.createFloatBuffer(M);
            for (int i = 0; i < M; i++) hostBias.put(i, (float) biasd[i]);
            clEnqueueWriteBuffer(cq, bufBias, true, 0, hostBias.rewind(), null, null);

            setKernelArgBuf(matvecKernel, 0, bufA);
            setKernelArgBuf(matvecKernel, 1, bufX);
            setKernelArgBuf(matvecKernel, 2, bufBias);
            setKernelArgBuf(matvecKernel, 3, bufY);
            clSetKernelArg1i(matvecKernel, 4, M);
            clSetKernelArg1i(matvecKernel, 5, N);

            PointerBuffer globalWorkSize = stack.mallocPointer(1);
            globalWorkSize.put(0, M);

            clEnqueueNDRangeKernel(cq, matvecKernel, 1, null, globalWorkSize.rewind(), null, null, null);
            clFinish(cq);

            FloatBuffer hostY = BufferUtils.createFloatBuffer(M);
            clEnqueueReadBuffer(cq, bufY, true, 0, hostY.rewind(), null, null);

            double[] result = new double[M];
            for (int i = 0; i < M; i++) result[i] = hostY.get(i);

            matvecCount++;
            totalGpuTimeNanos += System.nanoTime() - t0;
            return result;

        } catch (Exception e) {
            System.err.println("[GPU] matvec异常: " + e.getMessage());
            return matvecCPU(Ad, xd, biasd);
        }
    }

    public double[][] matmulGPU(double[][] Ad, double[][] Bd) {
        if (!isGpuAvailable()) return matmulCPU(Ad, Bd);

        long t0 = System.nanoTime();
        int M = Ad.length;
        int K = Ad[0].length;
        int N = Bd[0].length;

        try (MemoryStack stack = stackPush()) {
            IntBuffer errBuf = stack.mallocInt(1);
            long cq = ctx.getCommandQueue();

            long bufA = ensureBuffer(gpuBufA, capacityA, (long) M * K * 4, CL_MEM_READ_ONLY, errBuf);
            gpuBufA = bufA; capacityA = (long) M * K * 4;

            long bufB = ensureBuffer(gpuBufB, capacityB, (long) K * N * 4, CL_MEM_READ_ONLY, errBuf);
            gpuBufB = bufB; capacityB = (long) K * N * 4;

            long bufC = ensureBuffer(gpuBufC, capacityC, (long) M * N * 4, CL_MEM_WRITE_ONLY, errBuf);
            gpuBufC = bufC; capacityC = (long) M * N * 4;

            // 使用堆分配避免栈溢出
            FloatBuffer hostA = BufferUtils.createFloatBuffer(M * K);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < K; j++)
                    hostA.put(i * K + j, (float) Ad[i][j]);
            clEnqueueWriteBuffer(cq, bufA, true, 0, hostA.rewind(), null, null);

            FloatBuffer hostB = BufferUtils.createFloatBuffer(K * N);
            for (int i = 0; i < K; i++)
                for (int j = 0; j < N; j++)
                    hostB.put(i * N + j, (float) Bd[i][j]);
            clEnqueueWriteBuffer(cq, bufB, true, 0, hostB.rewind(), null, null);

            setKernelArgBuf(matmulKernel, 0, bufA);
            setKernelArgBuf(matmulKernel, 1, bufB);
            setKernelArgBuf(matmulKernel, 2, bufC);
            clSetKernelArg1i(matmulKernel, 3, M);
            clSetKernelArg1i(matmulKernel, 4, N);
            clSetKernelArg1i(matmulKernel, 5, K);

            PointerBuffer globalWorkSize = stack.mallocPointer(2);
            globalWorkSize.put(0, M);
            globalWorkSize.put(1, N);

            clEnqueueNDRangeKernel(cq, matmulKernel, 2, null, globalWorkSize.rewind(), null, null, null);
            clFinish(cq);

            FloatBuffer hostC = BufferUtils.createFloatBuffer(M * N);
            clEnqueueReadBuffer(cq, bufC, true, 0, hostC.rewind(), null, null);

            double[][] result = new double[M][N];
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result[i][j] = hostC.get(i * N + j);

            matmulCount++;
            totalGpuTimeNanos += System.nanoTime() - t0;
            return result;

        } catch (Exception e) {
            System.err.println("[GPU] matmul异常: " + e.getMessage());
            return matmulCPU(Ad, Bd);
        }
    }

    private long ensureBuffer(long existing, long existingCap, long needed, long flags, IntBuffer errBuf) {
        if (existing != 0 && existingCap >= needed) return existing;
        if (existing != 0) clReleaseMemObject(existing);
        return clCreateBuffer(ctx.getContextHandle(), flags, needed, errBuf);
    }

    private void setKernelArgBuf(long kernel, int index, long handle) {
        MemoryStack stack = MemoryStack.stackGet();
        int sp = stack.getPointer();
        try {
            LongBuffer buf = stack.mallocLong(1);
            buf.put(0, handle);
            clSetKernelArg(kernel, index, buf);
        } finally {
            stack.setPointer(sp);
        }
    }

    public double[] matvec(double[][] A, double[] x) {
        double[] bias = new double[A.length];
        return matvec(A, x, bias);
    }

    public double[] matvec(double[][] A, double[] x, double[] bias) {
        if (isGpuAvailable()) {
            return matvecGPU(A, x, bias);
        }
        return matvecCPU(A, x, bias);
    }

    public double[][] matmul(double[][] A, double[][] B) {
        if (isGpuAvailable()) {
            return matmulGPU(A, B);
        }
        return matmulCPU(A, B);
    }

    private double[] matvecCPU(double[][] A, double[] x, double[] bias) {
        int m = A.length;
        int n = A[0].length;
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = bias[i];
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            y[i] = sum;
        }
        return y;
    }

    private double[][] matmulCPU(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int k = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += A[i][p] * B[p][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    public double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] AT = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                AT[j][i] = A[i][j];
        return AT;
    }

    public double[][] elementwiseMul(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] * B[i][j];
        return C;
    }

    public double[][] addBias(double[][] A, double[] bias) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + bias[j];
        return C;
    }

    public double[][] activate(double[][] A, String activation) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = A[i][j];
                switch (activation.toUpperCase()) {
                    case "SIGMOID": C[i][j] = 1.0 / (1.0 + Math.exp(-val)); break;
                    case "RELU": C[i][j] = Math.max(0, val); break;
                    case "TANH": C[i][j] = Math.tanh(val); break;
                    default: C[i][j] = val;
                }
            }
        }
        return C;
    }

    public String getStats() {
        return String.format("GPU调用: matmul=%d, matvec=%d, 累计耗时=%.2f ms",
                matmulCount, matvecCount, totalGpuTimeNanos / 1_000_000.0);
    }

    public void shutdown() {
        if (matmulKernel != 0) clReleaseKernel(matmulKernel);
        if (matvecKernel != 0) clReleaseKernel(matvecKernel);
        if (activateKernel != 0) clReleaseKernel(activateKernel);
        if (matmulProgram != 0) clReleaseProgram(matmulProgram);
        if (matvecProgram != 0) clReleaseProgram(matvecProgram);
        if (activateProgram != 0) clReleaseProgram(activateProgram);
        if (gpuBufA != 0) clReleaseMemObject(gpuBufA);
        if (gpuBufB != 0) clReleaseMemObject(gpuBufB);
        if (gpuBufC != 0) clReleaseMemObject(gpuBufC);
        if (gpuBufX != 0) clReleaseMemObject(gpuBufX);
        if (gpuBufBias != 0) clReleaseMemObject(gpuBufBias);
        if (gpuBufY != 0) clReleaseMemObject(gpuBufY);
        System.out.println("[GPU] 资源已释放");
    }
}

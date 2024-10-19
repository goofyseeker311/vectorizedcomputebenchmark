package fi.jkauppa.vectorizedcomputebenchmark;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Random;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL12;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import static org.lwjgl.system.MemoryUtil.NULL;

public class VectorizedComputeBenchmark {
	private MemoryStack clStack = MemoryStack.stackPush();
	private final String clSource = 
			"kernel void sum(global const float *a, global const float *b, global float *answer) {"
			+ "unsigned int xid = get_global_id(0);"
			+ "answer[xid] = a[xid] * b[xid];"
			+ "}";	

	public VectorizedComputeBenchmark() {}

	public void run() {
		Random rand = new Random();
		int nc = 100000000; //1000M:1000000000, 100M:100000000, 1M:1000000, 1K:1000
		float[] a = new float[nc];
		float[] b = new float[nc];
		for (int i=0;i<nc;i++) {
			a[i] = rand.nextFloat();
			b[i] = rand.nextFloat();
		}
		System.out.println("init.");
		long stimestart = System.currentTimeMillis();
		float[] sc = new float[nc];
		for (int i=0;i<nc;i++) {
			sc[i] = a[i]*b[i];
		}
		long stimeend = System.currentTimeMillis();
		long stimedif = stimeend - stimestart;
		System.out.println("auto-vectorization: "+stimedif+"ms");
		float[] cc = new float[nc];
		Arrays.fill(cc, 0.0f);
		PointerBuffer clPlatforms = getClPlatforms();
		if (clPlatforms!=null) {
			System.out.println("jocl-vectorization: found "+clPlatforms.capacity()+" platforms");
			PointerBuffer clCtxProps = clStack.mallocPointer(3);
			clCtxProps.put(0, CL12.CL_CONTEXT_PLATFORM).put(2, 0);
			IntBuffer errcode_ret = clStack.callocInt(1);
			int[] errcode_int = new int[1]; 
			for (int p = 0; p < clPlatforms.capacity(); p++) {
				long platform = clPlatforms.get(p);
				clCtxProps.put(1, platform);
				String platformversion = getClPlatformInfo(platform, CL12.CL_PLATFORM_VERSION);
				System.out.println("jocl-vectorization: platform["+p+"] version: "+platformversion);
				PointerBuffer clDevices = getClDevices(platform);
				System.out.println("jocl-vectorization: platform["+p+"] devices: found "+clDevices.capacity()+" devices");
				for (int d = 0; d < clDevices.capacity(); d++) {
					long device = clDevices.get(d);
					String devicename = getClDeviceInfo(device, CL12.CL_DEVICE_NAME);
					System.out.println("jocl-vectorization: platform["+p+"] devices: device name: "+devicename);
					long context = CL12.clCreateContext(clCtxProps, device, (CLContextCallback)null, NULL, errcode_ret);
					if (errcode_ret.get(errcode_ret.position())==CL12.CL_SUCCESS) {
						System.out.println("jocl-vectorization: platform["+p+"] devices: device context successfully created");
						long clProgram = CL12.clCreateProgramWithSource(context, clSource, errcode_ret);
						CL12.clBuildProgram(clProgram, device, "", null, NULL);
						long clKernel = CL12.clCreateKernel(clProgram, "sum", errcode_ret);
						long clQueue = CL12.clCreateCommandQueue(context, device, CL12.CL_QUEUE_PROFILING_ENABLE, errcode_ret);
						long amem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_READ_ONLY, a, errcode_int);
						CL12.clEnqueueWriteBuffer(clQueue, amem, true, 0, a, null, null);
						long bmem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_READ_ONLY, b, errcode_int);
						CL12.clEnqueueWriteBuffer(clQueue, bmem, true, 0, b, null, null);
						long cmem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_WRITE_ONLY, cc, errcode_int);
						CL12.clSetKernelArg1p(clKernel, 0, amem);
						CL12.clSetKernelArg1p(clKernel, 1, bmem);
						CL12.clSetKernelArg1p(clKernel, 2, cmem);
						int dimensions = 1;
						PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(dimensions);
						globalWorkSize.put(0, nc);
						CL12.clFinish(clQueue);
						long ctimestart = System.currentTimeMillis();
						CL12.clEnqueueNDRangeKernel(clQueue, clKernel, dimensions, null, globalWorkSize, null, null, null);
						CL12.clFinish(clQueue);
						long ctimeend = System.currentTimeMillis();
						long ctimedif = ctimeend - ctimestart;
						FloatBuffer resultBuff = BufferUtils.createFloatBuffer(nc);
						CL12.clEnqueueReadBuffer(clQueue, cmem, true, 0, resultBuff, null, null);
						Arrays.fill(cc, 0.0f);
						resultBuff.rewind();
						resultBuff.get(0, cc);
						System.out.println("jocl-vectorization: platform["+p+"] compute: kernel vectorization: "+ctimedif+"ms");
					}
				}
			}
		} else {
			System.out.println("jocl-vectorization: platforms init failed");
		}
	}

	public static void main(String[] args) {
		VectorizedComputeBenchmark app = new VectorizedComputeBenchmark();
		app.run();
		System.out.println("done.");
	}

	private PointerBuffer getClPlatforms() {
		PointerBuffer platforms = null;
		IntBuffer pi = clStack.mallocInt(1);
		if (CL12.clGetPlatformIDs(null, pi)==CL12.CL_SUCCESS) {
			PointerBuffer clPlatforms = clStack.mallocPointer(pi.get(0));
			if (CL12.clGetPlatformIDs(clPlatforms, (IntBuffer)null)==CL12.CL_SUCCESS) {
				platforms = clPlatforms;
			}
		}
		return platforms;
	}

	private PointerBuffer getClDevices(long platform) {
		PointerBuffer devices = null;
		IntBuffer pi = clStack.mallocInt(1);
		if (CL12.clGetDeviceIDs(platform, CL12.CL_DEVICE_TYPE_ALL, null, pi)==CL12.CL_SUCCESS) {
			PointerBuffer pp = clStack.mallocPointer(pi.get(0));
			if (CL12.clGetDeviceIDs(platform, CL12.CL_DEVICE_TYPE_ALL, pp, (IntBuffer)null)==CL12.CL_SUCCESS) {
				devices = pp;
			}
		}
		return devices;
	}

	private String getClPlatformInfo(long platform, int param) {
		String platforminfo = null;
		PointerBuffer pp = clStack.mallocPointer(1);
		if (CL12.clGetPlatformInfo(platform, param, (ByteBuffer)null, pp)==CL12.CL_SUCCESS) {
			int bytes = (int)pp.get(0);
			ByteBuffer buffer = clStack.malloc(bytes);
			if (CL12.clGetPlatformInfo(platform, param, buffer, null)==CL12.CL_SUCCESS) {
				platforminfo = MemoryUtil.memUTF8(buffer, bytes - 1);
			}
		}
		return platforminfo;
	}

	private String getClDeviceInfo(long cl_device_id, int param_name) {
		String deviceinfo = null;
		PointerBuffer pp = clStack.mallocPointer(1);
		if (CL12.clGetDeviceInfo(cl_device_id, param_name, (ByteBuffer)null, pp)==CL12.CL_SUCCESS) {
			int bytes = (int)pp.get(0);
			ByteBuffer buffer = clStack.malloc(bytes);
			if (CL12.clGetDeviceInfo(cl_device_id, param_name, buffer, null)==CL12.CL_SUCCESS) {
				deviceinfo = MemoryUtil.memUTF8(buffer, bytes - 1);
			}

		}
		return deviceinfo;
	}

}

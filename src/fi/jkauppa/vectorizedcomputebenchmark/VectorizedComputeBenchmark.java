package fi.jkauppa.vectorizedcomputebenchmark;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL12;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import static org.lwjgl.system.MemoryUtil.NULL;

public class VectorizedComputeBenchmark {
	private MemoryStack clStack = MemoryStack.stackPush();
	private int nc;
	private int re;

	private final String clSource =
		"kernel void floatingop() {"
		+ "unsigned int xid = get_global_id(0);"
		+ "float id = (float)xid;"
		+ "float c = (id+1.23f)*id;"
		+ "}"
		+
		"kernel void scalarmult(global const float *a, global const float *b, global float *c) {"
		+ "unsigned int xid = get_global_id(0);"
		+ "c[xid] = a[xid] * b[xid];"
		+ "}"
		+
		"kernel void matrixmult(global const float *a, global const float *b, global float *c) {"
		+ "unsigned int xid = get_global_id(0);"
		+ "c[xid*4+0] = b[ 0]*a[xid*4+0] + b[ 1]*a[xid*4+1] + b[ 2]*a[xid*4+2] + b[ 3]*a[xid*4+3];"
		+ "c[xid*4+1] = b[ 4]*a[xid*4+0] + b[ 5]*a[xid*4+1] + b[ 6]*a[xid*4+2] + b[ 7]*a[xid*4+3];"
		+ "c[xid*4+2] = b[ 8]*a[xid*4+0] + b[ 9]*a[xid*4+1] + b[10]*a[xid*4+2] + b[11]*a[xid*4+3];"
		+ "c[xid*4+3] = b[12]*a[xid*4+0] + b[13]*a[xid*4+1] + b[14]*a[xid*4+2] + b[15]*a[xid*4+3];"
		+ "}";
	
	private void floatingOp(int size, int repeat) {
		for (int j=0;j<repeat;j++) {
			for (int i=0;i<size;i++) {
				float id = (float)i;
				@SuppressWarnings("unused")
				float c = (id+1.23f)*id;
			}
		}
	}
	
	private void scalarMult(float[] a, float[]b, float[] c, int size, int repeat) {
		for (int j=0;j<repeat;j++) {
			for (int i=0;i<size;i++) {
				c[i] = b[i]*a[i];
			}
		}
	}

	private void matrixMult(float[] a, float[]b, float[] c, int size, int repeat) {
		for (int j=0;j<repeat;j++) {
			for (int i=0;i<size;i++) {
				c[i*4+0] = b[ 0]*a[i*4+0] + b[ 1]*a[i*4+1] + b[ 2]*a[i*4+2] + b[ 3]*a[i*4+3];
				c[i*4+1] = b[ 4]*a[i*4+0] + b[ 5]*a[i*4+1] + b[ 6]*a[i*4+2] + b[ 7]*a[i*4+3];
				c[i*4+2] = b[ 8]*a[i*4+0] + b[ 9]*a[i*4+1] + b[10]*a[i*4+2] + b[11]*a[i*4+3];
				c[i*4+3] = b[12]*a[i*4+0] + b[13]*a[i*4+1] + b[14]*a[i*4+2] + b[15]*a[i*4+3];
			}
		}
	}

	public VectorizedComputeBenchmark(int vnc, int vre) {
		this.nc = vnc;
		this.re = vre;
	}

	public void run() {
		System.out.println("init.");
		System.out.println("Element count: "+this.nc+", Repeat count: "+this.re);
		Random rand = new Random();
		TreeMap<Long,Long> devicecontexts = initClDevices();
		Set<Long> devices = devicecontexts.keySet();
		
		float[] fa = {1.0f}; 
		long ftimestart = System.nanoTime();
		this.floatingOp(nc,re);
		long ftimeend = System.nanoTime();
		float ftimedif = (ftimeend-ftimestart)/(1000000.0f*re);
		System.out.println(String.format("%.4f",ftimedif).replace(",", ".")+"ms\t auto-vectorization: floatingop");
		for (Iterator<Long> d=devices.iterator();d.hasNext();) {
			Long device = d.next();
			Long context = devicecontexts.get(device);
			String devicename = getClDeviceInfo(device, CL12.CL_DEVICE_NAME);
			float ctimedif = runProgram(context, device, clSource, "floatingop", fa, fa, fa, nc, re)/re;
			System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t jocl-vectorization: floatingop: device: "+devicename);
		}
		
		float[] a = new float[nc];
		float[] b = new float[nc];
		for (int i=0;i<a.length;i++) {
			a[i] = rand.nextFloat();
			b[i] = rand.nextFloat();
		}
		float[] sc = new float[nc];
		long stimestart = System.nanoTime();
		this.scalarMult(a, b, sc, nc, re);
		long stimeend = System.nanoTime();
		float stimedif = (stimeend-stimestart)/(1000000.0f*re);
		System.out.println(String.format("%.4f",stimedif).replace(",", ".")+"ms\t auto-vectorization: scalarmult: ");
		float[] cc = new float[nc];
		for (Iterator<Long> d=devices.iterator();d.hasNext();) {
			Long device = d.next();
			Long context = devicecontexts.get(device);
			String devicename = getClDeviceInfo(device, CL12.CL_DEVICE_NAME);
			float ctimedif = runProgram(context, device, clSource, "scalarmult", a, b, cc, nc, re)/re;
			System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t jocl-vectorization: scalarmult: device: "+devicename);
		}
		
		int n4c = nc*4;
		float[] a2 = new float[n4c];
		for (int i=0;i<a2.length;i++) {
			a2[i] = rand.nextFloat();
		}
		float[] b2 = new float[16];
		for (int i=0;i<b2.length;i++) {
			b2[i] = rand.nextFloat();
		}
		float[] sc2 = new float[n4c];
		long s2timestart = System.nanoTime();
		this.matrixMult(a2, b2, sc2, nc, re);
		long s2timeend = System.nanoTime();
		float s2timedif = (s2timeend-s2timestart)/(1000000.0f*re);
		System.out.println(String.format("%.4f",s2timedif).replace(",", ".")+"ms\t auto-vectorization: matrixmult: ");
		float[] cc2 = new float[n4c];
		for (Iterator<Long> d=devices.iterator();d.hasNext();) {
			Long device = d.next();
			Long context = devicecontexts.get(device);
			String devicename = getClDeviceInfo(device, CL12.CL_DEVICE_NAME);
			float ctimedif = runProgram(context, device, clSource, "matrixmult", a2, b2, cc2, nc, re)/re;
			System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t jocl-vectorization: matrixmult: device: "+devicename);
		}
		
		System.out.println("done.");
	}

	public static void main(String[] args) {
		System.out.println("VectorizedComputeBenchmark v0.9.2");
		int nc = 100000000; //1000M:1000000000, 100M:100000000, 1M:1000000, 1K:1000
		int re = 1000;
		try {
			nc = Integer.parseInt(args[0]);
		} catch(Exception ex) {}
		try {
			re = Integer.parseInt(args[1]);
		} catch(Exception ex) {}
		VectorizedComputeBenchmark app = new VectorizedComputeBenchmark(nc,re);
		app.run();
		System.out.println("exit.");
	}

	private float runProgram(long context, long device, String source, String entry, float[] a, float[] b, float[] c, int size, int repeat) {
		float ctimedif = 0.0f;
		IntBuffer errcode_ret = clStack.callocInt(1);
		int[] errcode_int = new int[1];
		long clProgram = CL12.clCreateProgramWithSource(context, source, errcode_ret);
		CL12.clBuildProgram(clProgram, device, "", null, NULL);
		long clKernel = CL12.clCreateKernel(clProgram, entry, errcode_ret);
		long clQueue = CL12.clCreateCommandQueue(context, device, CL12.CL_QUEUE_PROFILING_ENABLE, errcode_ret);
		long amem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_READ_ONLY, a, errcode_int);
		CL12.clEnqueueWriteBuffer(clQueue, amem, true, 0, a, null, null);
		long bmem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_READ_ONLY, b, errcode_int);
		CL12.clEnqueueWriteBuffer(clQueue, bmem, true, 0, b, null, null);
		long cmem = CL12.clCreateBuffer(context, CL12.CL_MEM_COPY_HOST_PTR | CL12.CL_MEM_WRITE_ONLY, c, errcode_int);
		CL12.clSetKernelArg1p(clKernel, 0, amem);
		CL12.clSetKernelArg1p(clKernel, 1, bmem);
		CL12.clSetKernelArg1p(clKernel, 2, cmem);
		int dimensions = 1;
		PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(dimensions);
		globalWorkSize.put(0, size);
		CL12.clFinish(clQueue);
		PointerBuffer event = clStack.mallocPointer(1);
		PointerBuffer event2 = clStack.mallocPointer(1);
		if (CL12.clEnqueueNDRangeKernel(clQueue, clKernel, dimensions, null, globalWorkSize, null, null, event)==CL12.CL_SUCCESS) {
			for (int i=1;i<repeat;i++) {
				CL12.clEnqueueNDRangeKernel(clQueue, clKernel, dimensions, null, globalWorkSize, null, null, event2);
			}
			CL12.clWaitForEvents(event);
			CL12.clWaitForEvents(event2);
			long eventLong = event.get(0);
			long eventLong2 = event2.get(0);
			long[] ctimestart = {0};
			long[] ctimeend = {0};
			CL12.clGetEventProfilingInfo(eventLong, CL12.CL_PROFILING_COMMAND_START, ctimestart, (PointerBuffer)null);
			CL12.clGetEventProfilingInfo(eventLong2, CL12.CL_PROFILING_COMMAND_END, ctimeend, (PointerBuffer)null);
			ctimedif = (ctimeend[0]-ctimestart[0])/1000000.0f;
			FloatBuffer resultBuff = BufferUtils.createFloatBuffer(c.length);
			CL12.clEnqueueReadBuffer(clQueue, cmem, true, 0, resultBuff, null, null);
			CL12.clFinish(clQueue);
			Arrays.fill(c, 0.0f);
			resultBuff.rewind();
			resultBuff.get(0, c);
		}
		CL12.clReleaseMemObject(amem);
		CL12.clReleaseMemObject(bmem);
		CL12.clReleaseMemObject(cmem);
		CL12.clFinish(clQueue);
		CL12.clReleaseMemObject(clQueue);
		CL12.clReleaseKernel(clKernel);
		CL12.clReleaseProgram(clProgram);
		return ctimedif;
	}

	private TreeMap<Long,Long> initClDevices() {
		TreeMap<Long,Long> devices = new TreeMap<Long,Long>();
		PointerBuffer clPlatforms = getClPlatforms();
		if (clPlatforms!=null) {
			PointerBuffer clCtxProps = clStack.mallocPointer(3);
			clCtxProps.put(0, CL12.CL_CONTEXT_PLATFORM).put(2, 0);
			for (int p = 0; p < clPlatforms.capacity(); p++) {
				long platform = clPlatforms.get(p);
				clCtxProps.put(1, platform);
				PointerBuffer clDevices = getClDevices(platform);
				for (int d = 0; d < clDevices.capacity(); d++) {
					long device = clDevices.get(d);
					IntBuffer errcode_ret = clStack.callocInt(1);
					long context = CL12.clCreateContext(clCtxProps, device, (CLContextCallback)null, NULL, errcode_ret);
					if (errcode_ret.get(errcode_ret.position())==CL12.CL_SUCCESS) {
						devices.put(device, context);
					}
				}
			}
		}
		return devices;
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

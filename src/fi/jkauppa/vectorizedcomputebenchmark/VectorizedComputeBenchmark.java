package fi.jkauppa.vectorizedcomputebenchmark;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Random;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

public class VectorizedComputeBenchmark {
	private MemoryStack clStack = MemoryStack.stackPush();
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
		PointerBuffer clPlatforms = getClPlatforms();
		if (clPlatforms!=null) {
			System.out.println("jocl-vectorization: found "+clPlatforms.capacity()+" platforms");
			PointerBuffer clCtxProps = clStack.mallocPointer(3);
			clCtxProps.put(0, CL10.CL_CONTEXT_PLATFORM).put(2, 0);
			for (int p = 0; p < clPlatforms.capacity(); p++) {
				long platform = clPlatforms.get(p);
				clCtxProps.put(1, platform);
				String platformversion = getClPlatformInfo(platform, CL10.CL_PLATFORM_VERSION);
				System.out.println("jocl-vectorization: platform["+p+"] platformversion: "+platformversion);
				PointerBuffer devices = getClDevices(platform);
				for (int d = 0; d < devices.capacity(); d++) {
					long device = devices.get(d);
					String devicename = getClDeviceInfo(device, CL10.CL_DEVICE_NAME);
					System.out.println("jocl-vectorization: platform["+p+"] device name: "+devicename);
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
		if (CL10.clGetPlatformIDs(null, pi)==CL10.CL_SUCCESS) {
			PointerBuffer clPlatforms = clStack.mallocPointer(pi.get(0));
			if (CL10.clGetPlatformIDs(clPlatforms, (IntBuffer)null)==CL10.CL_SUCCESS) {
				platforms = clPlatforms;
			}
		}
		return platforms;
	}

	private PointerBuffer getClDevices(long platform) {
		PointerBuffer devices = null;
		IntBuffer pi = clStack.mallocInt(1);
		if (CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_ALL, null, pi)==CL10.CL_SUCCESS) {
			PointerBuffer pp = clStack.mallocPointer(pi.get(0));
			if (CL10.clGetDeviceIDs(platform, CL10.CL_DEVICE_TYPE_ALL, pp, (IntBuffer)null)==CL10.CL_SUCCESS) {
				devices = pp;
			}
		}
		return devices;
	}

	private String getClPlatformInfo(long platform, int param) {
		String platforminfo = null;
		PointerBuffer pp = clStack.mallocPointer(1);
		if (CL10.clGetPlatformInfo(platform, param, (ByteBuffer)null, pp)==CL10.CL_SUCCESS) {
			int bytes = (int)pp.get(0);
			ByteBuffer buffer = clStack.malloc(bytes);
			if (CL10.clGetPlatformInfo(platform, param, buffer, null)==CL10.CL_SUCCESS) {
				platforminfo = MemoryUtil.memUTF8(buffer, bytes - 1);
			}
		}
		return platforminfo;
	}

	private String getClDeviceInfo(long cl_device_id, int param_name) {
		String deviceinfo = null;
		PointerBuffer pp = clStack.mallocPointer(1);
		if (CL10.clGetDeviceInfo(cl_device_id, param_name, (ByteBuffer)null, pp)==CL10.CL_SUCCESS) {
			int bytes = (int)pp.get(0);
			ByteBuffer buffer = clStack.malloc(bytes);
			if (CL10.clGetDeviceInfo(cl_device_id, param_name, buffer, null)==CL10.CL_SUCCESS) {
				deviceinfo = MemoryUtil.memUTF8(buffer, bytes - 1);
			}

		}
		return deviceinfo;
	}

}

package fi.jkauppa.vectorizedcomputebenchmark;

import static org.lwjgl.opencl.CL10.clGetPlatformIDs;
import static org.lwjgl.opencl.CL10.clGetPlatformInfo;
import static org.lwjgl.system.MemoryStack.stackPush;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Random;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

public class VectorizedComputeBenchmark {
	private MemoryStack clStack = stackPush();
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
				if (platformversion!=null) {
					System.out.println("jocl-vectorization: platform["+p+"] platformversion: "+platformversion);
				} else {
					System.out.println("jocl-vectorization: platform["+p+"] platformversion failed");
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
		IntBuffer clPlatformsNum = clStack.mallocInt(1);
		if (CL10.clGetPlatformIDs(null, clPlatformsNum)==CL10.CL_SUCCESS) {
			PointerBuffer clPlatforms = clStack.mallocPointer(clPlatformsNum.get(0));
			if (clGetPlatformIDs(clPlatforms, (IntBuffer)null)==CL10.CL_SUCCESS) {
				platforms = clPlatforms;
			}
		}
		return platforms;
	}

	private String getClPlatformInfo(long platform, int param) {
		String platforminfo = null;
		PointerBuffer pp = clStack.mallocPointer(1);
		if (CL10.clGetPlatformInfo(platform, param, (ByteBuffer)null, pp)==CL10.CL_SUCCESS) {
			int bytes = (int)pp.get(0);
			ByteBuffer buffer = clStack.malloc(bytes);
			if (clGetPlatformInfo(platform, param, buffer, null)==CL10.CL_SUCCESS) {
				platforminfo = MemoryUtil.memUTF8(buffer, bytes - 1);
			}
		}
		return platforminfo;
	}
}

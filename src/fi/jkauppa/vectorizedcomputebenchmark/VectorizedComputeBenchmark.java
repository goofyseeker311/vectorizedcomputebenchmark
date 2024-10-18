package fi.jkauppa.vectorizedcomputebenchmark;

import static org.lwjgl.opencl.CL10.clGetPlatformIDs;
import static org.lwjgl.system.MemoryStack.stackPush;

import java.nio.IntBuffer;
import java.util.Random;

import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

public class VectorizedComputeBenchmark {
    public static void main(String[] args) {
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
    	MemoryStack clStack = stackPush();
    	IntBuffer clPlatformsNum = clStack.mallocInt(1);
    	if (CL10.clGetPlatformIDs(null, clPlatformsNum)==CL10.CL_SUCCESS) {
    		System.out.println("jocl-vectorization: found "+clPlatformsNum.get(0)+" platforms");
    		PointerBuffer clPlatforms = clStack.mallocPointer(clPlatformsNum.get(0));
    		if (clGetPlatformIDs(clPlatforms, (IntBuffer)null)==CL10.CL_SUCCESS) {
            	System.out.println("jocl-vectorization: platforms init success");
    		} else {
            	System.out.println("jocl-vectorization: platforms init failed");
    		}
    	} else {
        	System.out.println("jocl-vectorization: no platforms found");
    	}
    	System.out.println("done.");
    }
}

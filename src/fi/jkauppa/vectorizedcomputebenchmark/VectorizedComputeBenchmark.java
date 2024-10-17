package fi.jkauppa.vectorizedcomputebenchmark;

import java.util.Random;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class VectorizedComputeBenchmark {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    public static void main(String[] args) {
    	Random rand = new Random();
    	int nc = 100000000; //1000000000;
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
    	System.out.println("single threaded: "+stimedif+"ms");
    	long vtimestart = System.currentTimeMillis();
    	float[] vc = new float[nc];
    	vectorMultiply(a, b, vc);
    	long vtimeend = System.currentTimeMillis();
    	long vtimedif = vtimeend - vtimestart;
    	System.out.println("simd vectorized: "+vtimedif+"ms");
    	System.out.println("done.");
    }

    private static void vectorMultiply(float[] a, float[] b, float[] c) {
		int i = 0;
		for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
			FloatVector va = FloatVector.fromArray(SPECIES, a, i);
			FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
			FloatVector vc = va.mul(vb);
			vc.intoArray(c, i);
		}
		
		for (; i < a.length; i++) {
			c[i] = a[i] * b[i];
		}
	}
}

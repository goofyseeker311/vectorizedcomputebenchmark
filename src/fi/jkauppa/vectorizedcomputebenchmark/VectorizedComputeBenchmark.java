package fi.jkauppa.vectorizedcomputebenchmark;

import fi.jkauppa.vectorizedcomputebenchmark.ComputeLib.Device;

public class VectorizedComputeBenchmark {
	private ComputeLib computelib = new ComputeLib();
	private int de;
	private int nc;
	private int re;

	private final String clSource =
			"kernel void loopsmmult(global float *c) {"
			+ "unsigned int xid = get_global_id(0);"
			+ "float id = (float)xid;"
			+ "float loopsum = 0.0f;"
			+ "for (int y=0;y<72;y++) {"
			+   "for (int x=0;x<128;x++) {"
			+     "loopsum += (id+x)*y;"
			+   "}"
			+ "}"
			+ "c[xid] = loopsum;"
			+ "}";
	
	public VectorizedComputeBenchmark(int vde, int vnc, int vre) {
		this.de = vde;
		this.nc = vnc;
		this.re = vre;
	}

	public static void main(String[] args) {
		System.out.println("VectorizedComputeBenchmark v0.9.7");
		int de = 0;
		int nc = 100000000;
		int re = 100;
		try {de = Integer.parseInt(args[0]);} catch(Exception ex) {}
		try {nc = Integer.parseInt(args[1]);} catch(Exception ex) {}
		try {re = Integer.parseInt(args[2]);} catch(Exception ex) {}
		VectorizedComputeBenchmark app = new VectorizedComputeBenchmark(de,nc,re);
		app.run();
		System.out.println("exit.");
	}
	
	public void run() {
		System.out.println("init.");
		System.out.println("Element count: "+this.nc+", Repeat count: "+this.re);

		long device = computelib.devicelist[this.de];
		Device devicedata = computelib.devicemap.get(device);
		long queue = devicedata.queue;
		String devicename = devicedata.devicename;
		System.out.println("Using device["+de+"]: "+devicename);
		
		long[] cbuf = {computelib.createBuffer(device, nc)};
		long program = computelib.compileProgram(device, clSource);
		float ctimedif = computelib.runProgram(device, queue, program, "loopsmmult", cbuf, new int[]{0}, new int[]{nc}, re, true)/re;
		float tflops = (nc*3.0f*128.0f*72.0f*(1000.0f/ctimedif))/1000000000000.0f;
		System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t"+String.format("%.3f",tflops).replace(",", ".")+"tflops\t device: "+devicename);
		
		System.out.println("done.");
	}
}

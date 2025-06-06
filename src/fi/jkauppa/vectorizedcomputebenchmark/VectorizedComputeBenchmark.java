package fi.jkauppa.vectorizedcomputebenchmark;

import fi.jkauppa.vectorizedcomputebenchmark.ComputeLib.Device;

public class VectorizedComputeBenchmark {
	private ComputeLib computelib = new ComputeLib();
	private int de;
	private int nc;
	private int re;

	public VectorizedComputeBenchmark(int vde, int vnc, int vre) {
		this.de = vde;
		this.nc = vnc;
		this.re = vre;
	}

	public static void main(String[] args) {
		System.out.println("VectorizedComputeBenchmark v1.0.4");
		int de = 0;
		int nc = 100000000;
		int re = 1000;
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
		
		String clSource = ComputeLib.loadProgram("res/clprograms/benchmarks.cl", true);
		long program = computelib.compileProgram(device, clSource);
		
		if (true) {
			long[] cbuf = {computelib.createBuffer(device, nc)};
			float ctimedif = computelib.runProgram(device, queue, program, "loopsmmult", cbuf, new int[]{0}, new int[]{nc}, re, true)/re;
			float tflops = (nc*3.0f*128.0f*72.0f*(1000.0f/ctimedif))/1000000000000.0f;
			System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t"+String.format("%.3f",tflops).replace(",", ".")+"tflops\t device: "+devicename);
		}

		if (true) {
			long ncmem = computelib.createBuffer(device, 1);
			long[] cbuf = {computelib.createBuffer(device, nc*5), computelib.createBuffer(device, 1)};
			computelib.writeBufferi(device, queue, ncmem, new int[]{nc});
			float ctimedif = computelib.runProgram(device, queue, program, "loopsfill", cbuf, new int[]{0}, new int[]{nc}, re, true)/re;
			float gbytes = (nc*5.0f*4.0f*(1000.0f/ctimedif))/1000000000.0f;
			System.out.println(String.format("%.4f",ctimedif).replace(",", ".")+"ms\t"+String.format("%.3f",gbytes).replace(",", ".")+"GB/s\t device: "+devicename);
		}
		
		System.out.println("done.");
	}
}

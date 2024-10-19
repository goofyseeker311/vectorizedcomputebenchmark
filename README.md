# Vectorized Compute Benchmark

Java vectorized cpu/gpu tool for benchmarking generic math workloads.
Default example vectorization task is float array multiplication C=A*B.

```
C=A*B float[] mult :  1K   / 1M   / 100M
auto-vec-amd-r5800x:  0ms  / 3ms  / 94ms
simd-vec-amd-r5800x:  17ms / 43ms / 98ms
jocl-cpu-amd-r5800x:  0ms  / 1ms  / 46ms
jocl-gpu-nv-rtx3080:  0ms  / 1ms  / 56ms
```

Installation & compiling:
lwjgl 3.3.4 from https://www.lwjgl.org/customize
Eclipse IDE for Java Developers from https://www.eclipse.org/downloads/packages/

Running:
Java 23 JDK from https://www.oracle.com/java/technologies/downloads/

Run command from console:
```
java -jar vcb.jar
```

Enabling CPU as Windows OpenCL device: install w_opencl_runtime_p_2024.2.0.980.exe from
https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

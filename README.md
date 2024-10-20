# Vectorized Compute Benchmark

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

```
C=A*B float[] mult :  1K   / 1M   / 100M
auto-vec-amd-r5800x:  0ms  / 3ms  / 94ms
jocl-gpu-nv-rtx3080:  0ms  / 1ms  / 56ms
jocl-cpu-amd-r5800x:  0ms  / 1ms  / 46ms
```

```
C=A*B mat4*float4[]:  1K   / 1M   / 100M
auto-vec-amd-r5800x:  0ms  / 18ms / 426ms
jocl-gpu-nv-rtx3080:  0ms  / 3ms  / 214ms
jocl-cpu-amd-r5800x:  0ms  / 1ms  / 136ms
```

Compiling:
----------------

LWJGL/OpenCL 3.3.4: https://www.lwjgl.org/customize

Eclipse IDE for Java Developers: https://www.eclipse.org/downloads/packages/

Running:
----------------

Java 23 JDK: https://www.oracle.com/java/technologies/downloads/

Run command from console:
```
java -Xms8192M -jar vcb.jar
```

CPU OpenCL:
----------------
Enabling CPU as Windows OpenCL device:

install w_opencl_runtime_p_2024.2.0.980.exe:

https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

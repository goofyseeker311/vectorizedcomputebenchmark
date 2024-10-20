# Vectorized Compute Benchmark

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

```
C=A*B float[] mult :  1K   / 1M   / 10M   / 100M
auto-vec-amd-r5800x:  0ms  / 5ms  / 8ms   / 49ms
jocl-cpu-amd-r5800x:  0ms  / 1ms  / 5ms   / 47ms
jocl-gpu-nv-rtx3080:  0ms  / 1ms  / 7ms   / 53ms
```

```
C=A*B mat4*float4[]:  1K   / 1M   / 10M   / 100M
auto-vec-amd-r5800x:  0ms  / 18ms / 56ms  / 442ms
jocl-cpu-amd-r5800x:  0ms  / 2ms  / 14ms  / 136ms
jocl-gpu-nv-rtx3080:  0ms  / 3ms  / 22ms  / 197ms
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
java -Xms8192M -jar vcb.jar <num-elements=100000000>
```

CPU OpenCL:
----------------
Enabling CPU as Windows OpenCL device:

install w_opencl_runtime_p_2024.2.0.980.exe:

https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

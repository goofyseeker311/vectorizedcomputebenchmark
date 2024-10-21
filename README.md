# Vectorized Compute Benchmark

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

```
c=(id+1.0f)*id flop:     10 /   100 /    1K /   10K /  100K /    1M /   10M /  100M
auto-vec-amd-r5800x:        /       /  12us /       /       / 1.5ms /       / 1.5ms
jocl-cpu-amd-r5800x:        /       / 123us /       /       / 156us /       / 324us
jocl-gpu-nv-rtx3080:        /       /  42us /       /       /  48us /       / 313us
```

```
C=A*B float[] mult :     10 /   100 /    1K /   10K /  100K /    1M /   10M /  100M
auto-vec-amd-r5800x:    3us /   4us /  16us / 143us / 1.3ms / 5.1ms /   8ms /  49ms
jocl-cpu-amd-r5800x:  122us / 123us / 116us / 153us / 152us / 593us /   5ms /  47ms
jocl-gpu-nv-rtx3080:   56us /  56us /  55us /  57us /  83us / 766us /   7ms /  53ms
```

```
C=A*B mat4*float4[]:     10 /   100 /    1K /   10K /  100K /    1M /   10M /  100M
auto-vec-amd-r5800x:    8us /  54us / 275us / 2.3ms /  14ms /  17ms /  56ms / 442ms
jocl-cpu-amd-r5800x:   44us /  77us /  52us /  94us / 110us / 1.7ms /  14ms / 136ms
jocl-gpu-nv-rtx3080:   41us /  38us /  45us /  81us / 421us / 2.7ms /  22ms / 197ms
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


Reference Benchmark:
----------------
generated by: https://github.com/ProjectPhysX/OpenCL-Benchmark/
```
.-----------------------------------------------------------------------------.
|----------------.------------------------------------------------------------|
| Device ID    0 | NVIDIA GeForce RTX 3080                                    |
| Device ID    1 | AMD Ryzen 7 5800X 8-Core Processor                         |
|----------------'------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 0                                                          |
| Device Name    | NVIDIA GeForce RTX 3080                                    |
| Device Vendor  | NVIDIA Corporation                                         |
| Device Driver  | 565.90 (Windows)                                           |
| OpenCL Version | OpenCL C 1.2                                               |
| Compute Units  | 68 at 1710 MHz (8704 cores, 29.768 TFLOPs/s)               |
| Memory, Cache  | 10239 MB, 1904 KB global / 48 KB local                     |
| Buffer Limits  | 2559 MB global, 64 KB constant                             |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.518 TFLOPs/s (1/64) |
| FP32  compute                                        31.963 TFLOPs/s ( 1x ) |
| FP16  compute                                        32.943 TFLOPs/s ( 1x ) |
| INT64 compute                                         3.109  TIOPs/s (1/8 ) |
| INT32 compute                                        16.487  TIOPs/s (1/2 ) |
| INT16 compute                                        14.148  TIOPs/s (1/2 ) |
| INT8  compute                                        11.734  TIOPs/s (1/3 ) |
| Memory Bandwidth ( coalesced read      )                        699.86 GB/s |
| Memory Bandwidth ( coalesced      write)                        718.65 GB/s |
| Memory Bandwidth (misaligned read      )                        700.41 GB/s |
| Memory Bandwidth (misaligned      write)                        162.73 GB/s |
| PCIe   Bandwidth (send                 )                         13.88 GB/s |
| PCIe   Bandwidth (   receive           )                         13.62 GB/s |
| PCIe   Bandwidth (        bidirectional)            (Gen4 x16)   13.73 GB/s |
|-----------------------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 1                                                          |
| Device Name    | AMD Ryzen 7 5800X 8-Core Processor                         |
| Device Vendor  | Intel(R) Corporation                                       |
| Device Driver  | 2024.18.6.0.02_160000 (Windows)                            |
| OpenCL Version | OpenCL C 3.0                                               |
| Compute Units  | 16 at 0 MHz (8 cores, 0.000 TFLOPs/s)                      |
| Memory, Cache  | 130984 MB, 512 KB global / 32 KB local                     |
| Buffer Limits  | 130984 MB global, 128 KB constant                          |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.293 TFLOPs/s (1/64) |
| FP32  compute                                         0.284 TFLOPs/s (1/64) |
| FP16  compute                                          not supported        |
| INT64 compute                                         0.110  TIOPs/s (1/64) |
| INT32 compute                                         0.326  TIOPs/s (1/64) |
| INT16 compute                                         0.671  TIOPs/s (1/64) |
| INT8  compute                                         0.563  TIOPs/s (1/64) |
| Memory Bandwidth ( coalesced read      )                         35.91 GB/s |
| Memory Bandwidth ( coalesced      write)                         17.18 GB/s |
| Memory Bandwidth (misaligned read      )                         37.13 GB/s |
| Memory Bandwidth (misaligned      write)                         18.11 GB/s |
| PCIe   Bandwidth (send                 )                         21.08 GB/s |
| PCIe   Bandwidth (   receive           )                         19.52 GB/s |
| PCIe   Bandwidth (        bidirectional)            (Gen4 x16)   20.26 GB/s |
|-----------------------------------------------------------------------------|
|-----------------------------------------------------------------------------|
```

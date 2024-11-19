# Vectorized Compute Benchmark

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

  - Benchmark tflops workload: 72*128 nested loops of float32 sum(2x) and multiplication(1x) with result placement into memory.
  - Benchmark gzpixels workload: clear float5 rgba-z graphics buffer in memory.

```
Tflops elements:    10 /   100 /    1K /   10K /  100K /    1M /   10M /  100M
cpu-amd-r5800x:   24us / 108us / 166us / 1.3ms /  12ms / 111ms /  1.1s /   11s
gpu-nv-rtx3080:   34us /  35us /  36us /  34us / 137us / 1.3ms /  13ms / 125ms
```

```
GB/s elements:      10 /   100 /    1K /   10K /  100K /    1M /   10M /  100M
cpu-amd-r5800x:   30us /  29us /  31us /  15us /  31us / 190us /  11ms / 114ms
gpu-nv-rtx3080:   24us /  22us /  23us /  25us /  24us /  50us / 424us / 3.7ms
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
java -jar vcb.jar <num-elements=100000000> <num-repeats=1000>
```

CPU OpenCL:
----------------
Enabling CPU as Windows OpenCL device:

install w_opencl_runtime_p_2025.0.0.1166.exe:

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

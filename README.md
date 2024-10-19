# Vectorized Compute Benchmark
Vectorized compute benchmark tool

Java vectorized cpu/gpu tool for benchmarking generic math workloads.
Default example vectorization task is float array multiplication C=A*B.

```
                     1K   / 1M   / 100M
auto-vectors-5800x:  0ms  / 3ms  / 94ms
simd-vectors-5800x:  17ms / 43ms / 98ms
ocl-cpu-amd-r5800x:  0ms  / 1ms  / 46ms
ocl-gpu-nv-rtx3080:  0ms  / 1ms  / 56ms
```

installation: lwjgl 3.3.4 customize https://www.lwjgl.org/customize

running: java -jar vcb.jar

cpu as windows opencl device: install w_opencl_runtime_p_2024.2.0.980.exe from
https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

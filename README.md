# Vectorized Compute Benchmark
Vectorized compute benchmark tool

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

```
                     1K   / 1M   / 100M
auto-vectorization:  0ms  / 3ms  / 94ms
simd-vectorization:  17ms / 43ms / 98ms
jocl-vectorization:  TBD  / TBD  / TBD
```

installation: lwjgl 3.3.4 customize https://www.lwjgl.org/customize

running: add lwjgl/opencl jars in the class path

cpu as opencl device: install w_opencl_runtime_p_2024.2.0.980.exe from https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html

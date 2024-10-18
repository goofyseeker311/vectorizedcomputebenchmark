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

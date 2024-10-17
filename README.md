# Vectorized Compute Benchmark
Vectorized compute benchmark tool

Java vectorized cpu/gpu tool for benchmarking generic math workloads.

```
                     1K   / 1M   / 100M
auto-vectorization:  0ms  / 3ms  / 94ms
simd vectorization:  17ms / 43ms / 98ms
```

installation: openjdk-24-ea+20 from https://jdk.java.net/24/

running: --add-modules jdk.incubator.vector

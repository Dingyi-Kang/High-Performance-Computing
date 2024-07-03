![image](https://github.com/Dingyi-Kang/High-Performance-Computing/assets/81428296/8d77edb2-e4a0-4420-8191-6c019b9acf02)
![image](https://github.com/Dingyi-Kang/High-Performance-Computing/assets/81428296/66081a1b-aabb-4f81-b96a-9ff01b2bd712)


Related codes: 
  
            size_t BUF_PAD = 32;
            size_t CHUNK_SIZE = 2 * 8192;
            size_t nchunks = num_points / CHUNK_SIZE + (num_points % CHUNK_SIZE == 0 ? 0 : 1);
            std::vector<float> residuals(nchunks * BUF_PAD, 0.0);
            ///MARK: multi thread will write to common shared object -- residuals
            //residuals vector is allocated with a size of nchunks * BUF_PAD, 
            //This ensures that each chunk has a separate segment in the residuals vector, preventing multiple threads from writing to the same cache line.
    #pragma omp parallel for schedule(static, 32)
            for (int64_t chunk = 0; chunk < (int64_t)nchunks; ++chunk)
                for (size_t d = chunk * CHUNK_SIZE; d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d)
                    residuals[chunk * BUF_PAD] +=
                        math_utils::calc_distance(data + (d * dim), centers + (size_t)closest_center[d] * (size_t)dim, dim);
    
            for (size_t chunk = 0; chunk < nchunks; ++chunk)
                residual += residuals[chunk * BUF_PAD];
        }
    
        return residual;


The choice of `BUF_PAD` as 32 instead of 1 is to prevent false sharing more effectively. False sharing can severely degrade performance in a multi-threaded environment, and using a larger padding helps to mitigate this issue by ensuring that each thread writes to a different cache line.

### False Sharing Explanation

False sharing occurs when multiple threads write to different variables that reside on the same cache line. Since cache lines are the smallest unit of cache coherence, even if threads are writing to different variables, the entire cache line is marked dirty and needs to be written back to memory. This leads to unnecessary cache coherence traffic, which can degrade performance.

### Cache Line Size

- The typical cache line size on modern processors is 64 bytes.
- Each `float` in C++ is 4 bytes.

If `BUF_PAD` were set to 1, consecutive elements in the `residuals` array would likely fall within the same cache line, especially if `nchunks` is large. This could result in false sharing.

### Calculations with Different `BUF_PAD` Values

- **If `BUF_PAD` is 1**:
  - Consecutive entries in the `residuals` array would be `residuals[0]`, `residuals[1]`, `residuals[2]`, etc.
  - If these entries are updated by different threads, they could share the same cache line, leading to false sharing.

- **If `BUF_PAD` is 32**:
  - Consecutive entries in the `residuals` array would be `residuals[0]`, `residuals[32]`, `residuals[64]`, etc.
  - This ensures that each entry is on a different cache line, minimizing the chance of false sharing.

### Memory Layout with `BUF_PAD` = 32

With `BUF_PAD` set to 32:
- Each `residuals` entry that a thread writes to is spaced out by 32 `float`s.
- Given that each `float` is 4 bytes, 32 `float`s take 128 bytes, which ensures that they are on separate cache lines (assuming a 64-byte cache line).

### Example with Cache Line Size of 64 Bytes

Assume a 64-byte cache line and `BUF_PAD` of 32:
- Each entry in `residuals` is separated by 128 bytes (32 `float`s * 4 bytes/`float` = 128 bytes).
- Since 128 bytes is two cache lines, this ensures no two entries written by different threads fall on the same cache line.

### Code Snippet Impact

```cpp
#pragma omp parallel for schedule(static, 32)
for (int64_t chunk = 0; chunk < (int64_t)nchunks; ++chunk)
    for (size_t d = chunk * CHUNK_SIZE; d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d)
        residuals[chunk * BUF_PAD] +=
            math_utils::calc_distance(data + (d * dim), centers + (size_t)closest_center[d] * (size_t)dim, dim);

for (size_t chunk = 0; chunk < nchunks; ++chunk)
    residual += residuals[chunk * BUF_PAD];
```

- **Parallel Loop**: Each thread updates its own `residuals[chunk * BUF_PAD]`, spaced out by `BUF_PAD`.
- **Accumulation Loop**: The residuals are accumulated after the parallel loop.

### Conclusion

Using `BUF_PAD` of 32 instead of 1:
- Helps prevent false sharing by ensuring that each thread writes to a separate cache line.
- Improves performance by reducing unnecessary cache coherence traffic.

While `BUF_PAD` of 1 might be functionally correct, it would not provide the same performance benefits as a larger `BUF_PAD` like 32 in a multi-threaded environment.

### Detailed Explanation of `BUF_PAD` and Residual Calculation

The use of `BUF_PAD` and the calculation of residuals in the `lloyds_iter` function are designed to optimize memory access and computation in parallel processing. Let's break down their usage and purpose.

### `BUF_PAD`

1. **Definition and Purpose**
   ```cpp
   size_t BUF_PAD = 32;
   ```
   - `BUF_PAD` is a buffer padding size used to ensure that each thread writes to a distinct memory location to avoid false sharing.
   - False sharing occurs when multiple threads update variables that reside on the same cache line, leading to performance degradation.

2. **Usage in Residuals Calculation**
   ```cpp
   size_t nchunks = num_points / CHUNK_SIZE + (num_points % CHUNK_SIZE == 0 ? 0 : 1);
   std::vector<float> residuals(nchunks * BUF_PAD, 0.0);
   ```
   - The `residuals` vector is allocated with a size of `nchunks * BUF_PAD`, where `nchunks` is the number of chunks the data is divided into.
   - This ensures that each chunk has a separate segment in the `residuals` vector, preventing multiple threads from writing to the same cache line.

### Residual Calculation

1. **Parallel Processing Setup**
   ```cpp
   #pragma omp parallel for schedule(static, 32)
   for (int64_t chunk = 0; chunk < (int64_t)nchunks; ++chunk)
   {
       for (size_t d = chunk * CHUNK_SIZE; d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d)
       {
           residuals[chunk * BUF_PAD] +=
               math_utils::calc_distance(data + (d * dim), centers + (size_t)closest_center[d] * (size_t)dim, dim);
       }
   }
   ```
   - The outer loop iterates over chunks of data points, each chunk being processed by a separate thread.
   - The inner loop iterates over the data points within the current chunk.
   - For each data point `d`, the distance to its closest center is calculated and added to the corresponding entry in the `residuals` vector.

2. **Distance Calculation**
   ```cpp
   residuals[chunk * BUF_PAD] +=
       math_utils::calc_distance(data + (d * dim), centers + (size_t)closest_center[d] * (size_t)dim, dim);
   ```
   - `data + (d * dim)`: Pointer to the `d`-th data point.
   - `centers + (size_t)closest_center[d] * (size_t)dim`: Pointer to the center that is closest to the `d`-th data point.
   - `math_utils::calc_distance(..., ..., dim)`: Function that calculates the Euclidean distance between the data point and the closest center.
   - `residuals[chunk * BUF_PAD] += ...`: Adds the calculated distance to the corresponding entry in the `residuals` vector. The index `chunk * BUF_PAD` ensures that each chunk writes to a separate location in the `residuals` vector.

3. **Accumulating the Residuals**
   ```cpp
   for (size_t chunk = 0; chunk < nchunks; ++chunk)
       residual += residuals[chunk * BUF_PAD];
   ```
   - After the parallel computation, the residuals from all chunks are accumulated to compute the total residual.
   - Each `residuals[chunk * BUF_PAD]` contains the sum of distances for the corresponding chunk, ensuring that there is no overlap or interference between threads.

### Summary

- **`BUF_PAD`**: Used to pad the `residuals` array to prevent false sharing by ensuring each thread writes to a separate cache line.
- **`residuals[chunk * BUF_PAD] += ...`**: Accumulates the sum of distances for data points within a chunk to the corresponding entry in the `residuals` array.
- **Purpose**: Optimizes memory access and computation in parallel processing, ensuring efficient and correct calculation of the residuals.

This approach ensures that the computation is both efficient (due to parallel processing) and correct (by avoiding false sharing and ensuring accurate accumulation of distances).

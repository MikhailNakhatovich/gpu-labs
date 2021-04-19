# Lab2

Square matrix multiplication with CUDA and shared memory using.

## Results

| Matrix dimension | CPU time, s | GPU time, s | GPU with shared memory time, s | Maximum deviation between CPU and GPU results | Maximum deviation between CPU and GPU with shared memory results |
| --- | --- | --- | --- | --- | --- |
| 1000 | 1.0337 | 0.021543 | 0.019155 | 4.00178e-11 | 4.00178e-11 |
| 2000 | 11.3319 | 0.157971 | 0.123619 | 6.54836e-11 | 6.54836e-11 |
| 5000 | 613.231 | 2.33265 | 1.71748 | 1.23691e-10 | 1.23691e-10 |

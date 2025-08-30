# VisionGuard Benchmark Demo Results

## CPU Performance Table

| Quant       | Device   | Limit     |   FPS |   Latency (ms) |   CPU (%) |   RAM (MB) |   GPU (%) |   GPU Shared Memory (GB) |   NPU (%) |   NPU Memory (MB) |   Intel GPU (%) |   Intel GPU Memory (MB) |
|:------------|:---------|:----------|------:|---------------:|----------:|-----------:|----------:|-------------------------:|----------:|------------------:|----------------:|------------------------:|
| FP32        | CPU      | 10 - FPS  |  17.8 |           16   |       3.3 |      215.7 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| FP32        | CPU      | 25 - FPS  |  20.4 |            8.8 |      11.5 |      402.2 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| FP32        | CPU      | Max - FPS |   9.8 |            4.3 |       4.5 |      335.2 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| FP16        | CPU      | 10 - FPS  |  26.9 |           19.9 |       2.6 |      376.4 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| FP16        | CPU      | 25 - FPS  |  13.6 |           28.3 |       4.6 |      245.5 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| FP16        | CPU      | Max - FPS |  26   |           13.2 |       7.1 |      495.1 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | CPU      | 10 - FPS  |  31   |           15.1 |      12.8 |      517   |       nan |                      nan |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | CPU      | 25 - FPS  |   7.1 |           24.4 |      16.2 |      231.9 |       nan |                      nan |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | CPU      | Max - FPS |  21   |            5.2 |       3.3 |      323.7 |       nan |                      nan |       nan |               nan |             nan |                     nan |

## GPU Performance Table

| Quant       | Device   | Limit     |   FPS |   Latency (ms) |   CPU (%) |   RAM (MB) |   GPU (%) |   GPU Shared Memory (GB) |   NPU (%) |   NPU Memory (MB) |   Intel GPU (%) |   Intel GPU Memory (MB) |
|:------------|:---------|:----------|------:|---------------:|----------:|-----------:|----------:|-------------------------:|----------:|------------------:|----------------:|------------------------:|
| FP32        | GPU      | 10 - FPS  |  27.1 |            7.3 |       1.9 |      470.7 |      67.8 |                      2.1 |       nan |               nan |             nan |                     nan |
| FP32        | GPU      | 25 - FPS  |  12.4 |           25.8 |      16.5 |      450.5 |      31.6 |                      2   |       nan |               nan |             nan |                     nan |
| FP32        | GPU      | Max - FPS |  13.2 |           13.9 |       5.2 |      239.4 |      59.3 |                      2.2 |       nan |               nan |             nan |                     nan |
| FP16        | GPU      | 10 - FPS  |  25.4 |           15.8 |       7.6 |      547.1 |      35.7 |                      2.1 |       nan |               nan |             nan |                     nan |
| FP16        | GPU      | 25 - FPS  |  11.2 |           20.6 |      17.5 |      332   |      66.2 |                      2.2 |       nan |               nan |             nan |                     nan |
| FP16        | GPU      | Max - FPS |  23.8 |           25.1 |       5.1 |      205.3 |      34.2 |                      2.1 |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | GPU      | 10 - FPS  |  14.5 |           26   |      14.5 |      257.5 |      47.8 |                      2   |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | GPU      | 25 - FPS  |  19.2 |           24.7 |      13.2 |      568.4 |      23   |                      2   |       nan |               nan |             nan |                     nan |
| INT-8 FP-16 | GPU      | Max - FPS |  17   |           28.1 |       5.7 |      309.8 |      33.6 |                      2   |       nan |               nan |             nan |                     nan |

## NPU Performance Table

| Quant       | Device   | Limit     |   FPS |   Latency (ms) |   CPU (%) |   RAM (MB) |   GPU (%) |   GPU Shared Memory (GB) |   NPU (%) |   NPU Memory (MB) |   Intel GPU (%) |   Intel GPU Memory (MB) |
|:------------|:---------|:----------|------:|---------------:|----------:|-----------:|----------:|-------------------------:|----------:|------------------:|----------------:|------------------------:|
| FP32        | NPU      | 10 - FPS  |  22.4 |            7.9 |       5   |      429.4 |       nan |                      nan |      18.7 |             193.1 |             nan |                     nan |
| FP32        | NPU      | 25 - FPS  |   7.2 |           11.2 |      12.3 |      555.3 |       nan |                      nan |       6.5 |             201.5 |             nan |                     nan |
| FP32        | NPU      | Max - FPS |  18.4 |           17   |      12.9 |      449.5 |       nan |                      nan |      26.5 |             189.6 |             nan |                     nan |
| FP16        | NPU      | 10 - FPS  |  19.6 |           21.1 |       4.3 |      280.3 |       nan |                      nan |      14.4 |             205.8 |             nan |                     nan |
| FP16        | NPU      | 25 - FPS  |  19.8 |           24.7 |      18.8 |      323.8 |       nan |                      nan |       8   |             185.2 |             nan |                     nan |
| FP16        | NPU      | Max - FPS |   9.7 |           28.3 |       4.7 |      324.3 |       nan |                      nan |       9.1 |             186   |             nan |                     nan |
| INT-8 FP-16 | NPU      | 10 - FPS  |  16.2 |           28.3 |      12.5 |      465.8 |       nan |                      nan |      16.4 |             216.3 |             nan |                     nan |
| INT-8 FP-16 | NPU      | 25 - FPS  |  26.5 |           17   |       4.5 |      569.2 |       nan |                      nan |      29.7 |             183.7 |             nan |                     nan |
| INT-8 FP-16 | NPU      | Max - FPS |  15.6 |            6.5 |       6.7 |      289.9 |       nan |                      nan |       7   |             193.9 |             nan |                     nan |

## IGPU Performance Table

| Quant       | Device   | Limit     |   FPS |   Latency (ms) |   CPU (%) |   RAM (MB) |   GPU (%) |   GPU Shared Memory (GB) |   NPU (%) |   NPU Memory (MB) |   Intel GPU (%) |   Intel GPU Memory (MB) |
|:------------|:---------|:----------|------:|---------------:|----------:|-----------:|----------:|-------------------------:|----------:|------------------:|----------------:|------------------------:|
| FP32        | IGPU     | 10 - FPS  |  26.5 |           12.6 |      13.8 |      228.9 |       nan |                      nan |       nan |               nan |             4.8 |                   313.6 |
| FP32        | IGPU     | 25 - FPS  |  22.5 |           28.2 |       5.8 |      580.1 |       nan |                      nan |       nan |               nan |            11.4 |                   301.8 |
| FP32        | IGPU     | Max - FPS |  28.7 |           28.1 |      18.4 |      258.5 |       nan |                      nan |       nan |               nan |             8.4 |                   301   |
| FP16        | IGPU     | 10 - FPS  |  15   |           28.4 |       9.3 |      575.5 |       nan |                      nan |       nan |               nan |            24.8 |                   301.9 |
| FP16        | IGPU     | 25 - FPS  |  16.4 |           11.5 |       1   |      300.3 |       nan |                      nan |       nan |               nan |            15.1 |                   286.7 |
| FP16        | IGPU     | Max - FPS |  25.2 |            5.9 |       4.7 |      287.3 |       nan |                      nan |       nan |               nan |            18.8 |                   340.7 |
| INT-8 FP-16 | IGPU     | 10 - FPS  |  12.2 |           16.9 |      12.8 |      551.5 |       nan |                      nan |       nan |               nan |            11.4 |                   414.9 |
| INT-8 FP-16 | IGPU     | 25 - FPS  |  25.9 |           29.3 |       1.2 |      444.9 |       nan |                      nan |       nan |               nan |             5.6 |                   400.2 |
| INT-8 FP-16 | IGPU     | Max - FPS |  22   |           16.2 |      15   |      418   |       nan |                      nan |       nan |               nan |             2.6 |                   310.9 |

## AUTO Performance Table

| Quant       | Device   | Limit     |   FPS |   Latency (ms) |   CPU (%) |   RAM (MB) |   GPU (%) |   GPU Shared Memory (GB) |   NPU (%) |   NPU Memory (MB) |   Intel GPU (%) |   Intel GPU Memory (MB) |
|:------------|:---------|:----------|------:|---------------:|----------:|-----------:|----------:|-------------------------:|----------:|------------------:|----------------:|------------------------:|
| FP32        | AUTO     | 10 - FPS  |  10.2 |           25.1 |       6.3 |      406.2 |       nan |                      nan |       nan |               nan |            18.2 |                   313.2 |
| FP32        | AUTO     | 25 - FPS  |  18.8 |           17.9 |      15.2 |      505.2 |       nan |                      nan |       nan |               nan |            21.1 |                   373.6 |
| FP32        | AUTO     | Max - FPS |   9.9 |           21.7 |      16   |      249.7 |       nan |                      nan |       nan |               nan |            27.7 |                   289   |
| FP16        | AUTO     | 10 - FPS  |  24.5 |            6.6 |       3.8 |      228.1 |       nan |                      nan |       nan |               nan |            22.9 |                   398.5 |
| FP16        | AUTO     | 25 - FPS  |  27.4 |            8.4 |       1.2 |      380.3 |       nan |                      nan |       nan |               nan |            16   |                   400.4 |
| FP16        | AUTO     | Max - FPS |  17.4 |            7.2 |       8.1 |      543.6 |       nan |                      nan |       nan |               nan |             6.4 |                   415.5 |
| INT-8 FP-16 | AUTO     | 10 - FPS  |  15.6 |           16.7 |      13.4 |      298.2 |       nan |                      nan |       nan |               nan |             7   |                   292.3 |
| INT-8 FP-16 | AUTO     | 25 - FPS  |  12   |           10.5 |       8.8 |      479   |       nan |                      nan |       nan |               nan |            27.3 |                   389   |
| INT-8 FP-16 | AUTO     | Max - FPS |   9.3 |           12.2 |       3.3 |      266.1 |       nan |                      nan |       nan |               nan |             7.2 |                   337.1 |

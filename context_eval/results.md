# Context Management Strategy Comparison

| Strategy | Detail recalled correctly | Avg. input tokens/run | Avg. output tokens/run | Avg. latency |
|---|---|---|---|---|
| sliding_window | 0/10 | 499 | 39 | 0.3s |
| observation_masking | 10/10 | 819 | 26 | 7.54s |
| recursive_summarization | 10/10 | 480 | 329 | 23.87s |
| zone_based_pruning | 10/10 | 1168 | 26 | 13.15s |

# Primary Investigation

Does adaptive prefix tuning on encoder-only models (such as BERT) improve performance?

1. Accuracy
2. Compute
    - training time (in hours)
    - FLOPs (floating point operations per second) (maybe)

How to determine if this is the case? [using prefix length of 4, 8, 16, 32]
- Train baseline (fine-tune only last layer)
- Train prefix-tuned
- Train gated prefix-tuned

and then compare

Using 3 datasets: Multi-NLI, AGNews, and one of the small datasets (Treebank, COLA, or Scitail)
- 1 person take 1 dataset
- since the large datasets might be quite big, feel free to downsample the dataset

# Reseach Questions
 
1. How many prefixes are needed for prefix tuning?
2. Does prefix tuning improve performance over baselines? What about adaptive (gated) prefix tuning?
3. Compare compute / time stuff between 3 model types (baseline, prefix, adaptive)? Is it worth it?

Maybe combine 1 and 2
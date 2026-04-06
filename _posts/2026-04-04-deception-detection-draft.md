---
title: "Deception Detection in Qwen 3.5"
published: false
summary: "A compact draft on where deception signal concentrates in Qwen 3.5-4B and why a narrower layer range outperforms broader probe coverage."
topics:
  - Linear probes
  - Layer analysis
---

Key findings to justify layer selection:

- Deception is localized: layers 10-15 (31-47% depth) carry the strongest cross-domain deception signal in Qwen 3.5-4B.
- More layers is not better: the 32-layer combined probe performed worse on honest-vs-deceptive than a focused 7-layer probe, indicating noise from uninformative layers.
- Early layers (0-7) are uninformative: near-chance AUROC and no usable deception representation.
- Late layers (24-31) show declining signal: the model is transitioning from abstract representation toward next-token prediction.
- The sweet spot sits earlier than the conventional middle-half heuristic: the default `range(8, 24)` includes many suboptimal layers, while the actual peak is `range(10, 16)`.
- Different metrics favor slightly different layers: layer 14 best discriminates honest from deceptive, while layers 13 and 15 best separate deception from neutral text. A multi-layer probe covering 10-15 captures both strengths.

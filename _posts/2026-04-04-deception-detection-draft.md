---
layout: default
title: "Deception Detection in Qwen 3.5"
---

<nav aria-label="Site navigation" style="margin: 0 0 1.5rem 0; padding: 0.7rem 0.9rem; border: 1px solid #d0d0d0; border-radius: 6px; background: #f7f7f7; font-size: 0.95rem; font-weight: 600; line-height: 1.2;">
  <a href="{{ '/' | relative_url }}">Home</a>
  <span style="color: #888;"> | </span>
  <a href="{{ '/writing/' | relative_url }}">Writing</a>
  <span style="color: #888;"> | </span>
  <a href="https://github.com/neveratdennys">GitHub</a>
</nav>

Key findings to justify layer selection:

Deception is localized — layers 10-15 (31-47% depth) carry the strongest cross-domain deception signal in Qwen3.5-4B
More layers ≠ better — the 32-layer combined probe performed worse on honest-vs-deceptive than a focused 7-layer probe, indicating noise from uninformative layers
Early layers (0-7) are uninformative — near-chance AUROC, no usable deception representation
Late layers (24-31) show declining signal — the model transitions from abstract representation toward next-token prediction
The "sweet spot" sits earlier than the conventional middle-half heuristic — the default range(8, 24) includes many suboptimal layers; the actual peak is range(10, 16)
Different metrics favor slightly different layers — layer 14 best discriminates honest from deceptive; layers 13/15 best separate deception from neutral text. A multi-layer probe covering 10-15 captures both strengths.

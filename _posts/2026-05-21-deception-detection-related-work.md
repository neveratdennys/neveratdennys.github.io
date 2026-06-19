---
title: "Related Work: Deception Probes, One Year After Apollo"
published: true
summary: "An survey of follow up work to Apollo's deception probe paper published between February 2025 and April 2026. Where my Qwen thread overlaps with what's been done, where it disagrees, and where it still looks open."
topics:
  - Interpretability
  - Related work
  - Qwen 3.5
---

After wrapping the omission probe stage, I stepped back and did what I should have done first: checked what else has been published along this line since Apollo Research's [February 2025 paper](https://arxiv.org/abs/2502.03407). After some searches, the goal here is to map the landscape and flag which parts of my work have live neighbors in the literature.

## Live scoop risks

Three papers sit closest to what I've been doing.

**Nordby, Pais, Parrack — "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling"** ([arXiv 2604.13386](https://arxiv.org/abs/2604.13386), April 2026). Tests 12 models across Llama, Mistral, and **Qwen (0.5B, 3B, 14B, 72B)**. Runs a layer sweep, proposes multi-layer ensembling, characterises the deception feature as slowly rotating across depth. This is the biggest direct overlap with my Qwen replication and 32-layer sweep. Their headline result puts optimal probe layers in the **latter two-thirds of depth (mean ~65%)**, which sits in tension with my layers 11-14 commission peak (around 43% depth) on Qwen 3.5-4B. An earlier draft of this survey argued the tension partially resolved by conditioning on probe type, but under the canonical b=4 re-derivation (see the [batch size post]({{ '/writing/deception-detection-batch-size-kernel-dispatch/' | relative_url }})) the omission probe is flat across depth rather than peaking late, so the "omission picks up the latter two-thirds" reconciliation doesn't actually run. The remaining open questions are (a) whether depth fractions are even comparable across model scales (Nordby et al. push back on casual cross-scale reasoning of this kind), (b) whether the 4B scale behaves differently from their larger Qwen variants, and (c) whether Qwen 3.5's hybrid Gated DeltaNet + Attention architecture (which their sweep doesn't isolate) changes the depth story vs pure-attention Qwen. The disagreement is publishable as a live question rather than a problem.

**Wang et al. — "Beyond Prompt-Induced Lies"** ([arXiv 2508.06361](https://arxiv.org/abs/2508.06361), ICLR 2026 Oral). Evaluates 16 LLMs including multiple Qwen variants on a graph reachability task and explicitly distinguishes **fabrication (commission-like) and concealment (omission-like)** as separate derivable scores. This is the closest prior art to my commission-vs-omission framing. The key difference: Wang et al. measure behavior, not activations, and make no claim about representational distinctness inside the model. The conceptual split is in the air and published in an oral-accepted venue, so my work is the activation level complement to their behavioral result, not first-mover on the split itself.

**Natarajan et al. — "Building Better Deception Probes Using Targeted Instruction Pairs"** ([arXiv 2602.01425](https://arxiv.org/abs/2602.01425), February 2026). Gemma-2-9B only. Builds a taxonomy of deception types (white lies, concealment, evasion) and trains targeted probes, reporting +20.5% AUC improvements over untargeted baselines. Closest to my "multiple probes for different deception types" direction. Their categories are taxonomic and behavioral (white lie / concealment / evasion) rather than representational (commission / omission), and they report that layer choice explains only 2.7% of variance in their setup, a weaker role for layers than I find on Qwen. 

## Other partial overlaps

**Laurito et al. — "Liars' Bench"** ([arXiv 2511.16035](https://arxiv.org/abs/2511.16035), November 2025). 72,863 on-policy lie/honest examples from Llama-3.3-70B, **Qwen-2.5-72B**, Mistral-Small-24B, and Gemma-3-27B. Reports mean probes averaging balanced accuracy 0.54, close to chance, and identifies that probes systematically fail on certain lie types. Documents the general cross-domain failure pattern I hit, though without the specific inverted-direction signature or the commission/omission split.

**"Caught in the Act"** ([arXiv 2508.19505](https://arxiv.org/abs/2508.19505), September 2025). Probes Llama, **Qwen (1.5B–14B)**, and DeepSeek-r1. Runs an explicit layer-wise sweep and reports a three stage pattern: near random early, peak mid, slight decline late. Qualitatively consistent with my commission-probe shape, though they don't carve by deception type.

**Smith, Chughtai, Nanda — "Difficulties with Evaluating a Deception Detector for AIs"** ([arXiv 2511.22662](https://arxiv.org/abs/2511.22662), November 2025). Methodological critique from Apollo/DeepMind-adjacent authors arguing cross domain transfer is under tested and labels are underdetermined. Friendly to my framing. This is the kind of paper my insider-trading result is a concrete instance of.

**Parrack, Hex, Nardo — "Trusted monitoring, but with deception probes"** ([LessWrong, July 2025](https://www.lesswrong.com/posts/eaEqAzGN3uJfpfGoc/trusted-monitoring-but-with-deception-probes)). Apollo affiliated benchmarking of their two probes across five deception datasets, reporting 97% TPR at 1% FPR and showing one probe generalizes to backdoored code. Llama-3.3-70B only. No Qwen, no category split.

**Boxo et al. — "Benchmarking Deception Probes via Black-to-White Performance Boosts"** ([arXiv 2507.12691](https://arxiv.org/abs/2507.12691), January 2026). Compares probe-augmented white box monitoring against black-box. Finds weak but encouraging gains. Evaluation methodology focus, not categorization-focus.

**Fang et al. — "MM-DeceptionBench"** ([arXiv 2512.00349](https://arxiv.org/abs/2512.00349), December 2025). Multimodal, game theory framing. Names **six deception categories explicitly, including deliberate omission and fabrication**. Behavioral benchmark, not probe-based. Reinforces that omission as its own category is a thread multiple groups are pulling on.

**"Pressure-Testing Deception Probes"** ([arXiv 2605.27958](https://arxiv.org/abs/2605.27958), May 2026). Traces the cross domain transfer failure geometrically, finding that a single linear direction does not capture deception across types, and fixes it with style augmented single probes using several directions rather than per category probes. A direct alternative fix to the same failure my omission probe addresses, worth comparing against rather than just citing.

**"Detecting High-Stakes Interactions with Activation Probes"** ([arXiv 2506.10805](https://arxiv.org/abs/2506.10805), June 2025). Trains single concept probes for high stakes situations and proposes combining concept specific probes (deception, sycophancy, stakes) as future work. Adjacent to my multi probe direction, but the combination is across different concepts rather than within deception sub types.

## Where my angle still looks open

1. **Using human deception neuroscience as a categorization framework** for activation probes appears to be unclaimed, based on my searches rather than an exhaustive review. The neuro AI survey literature I found goes the other direction (using LLMs as brain models).
2. **The specific inverted-AUROC-0.458 failure mode** on insider trading with a paired-facts (RepE) probe on Qwen, and the per-domain recovery from a dedicated omission probe. Documented in [post 2]({{ '/writing/deception-detection-two-signals/' | relative_url }}) and a forthcoming multi probe POC writeup. Liars' Bench documents broad probe failures, but I haven't found a prior report of this specific directional inversion for a RepE commission probe on insider trading at this model scale.

## What this means for the project

The Phase 2 multi probe POC is a forthcoming separate post, "Two Probes, One Confusion Matrix." It covers per-probe layer selection, the combined-LR vs score-averaging asymmetry on omission, and the +0.36 AUROC recovery on insider trading. (I'll link it here once it's published.)

The activation level framing of commission vs omission still looks unclaimed in the literature, but Wang et al. (ICLR 2026 Oral) and Natarajan et al. (February 2026) mean the conceptual split is in the air. The Nordby et al. layer-depth disagreement is worth treating as a live question. The right comparison would be a controlled study on larger Qwen variants and on pure-attention models, which is outside my current scope.

This list isn't exhaustive. I'll keep referring to adjacent work where relevant and add new studies as they appear.
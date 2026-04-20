---
title: "Related Work: Deception Probes, One Year After Apollo"
published: false
summary: "An initial survey of follow-up work to Apollo's deception-probe paper published between February 2025 and April 2026. Where my Qwen thread overlaps with what's been done, where it disagrees, and where it still looks open."
topics:
  - Interpretability
  - Related work
  - Qwen 3.5
---

After wrapping the omission-probe stage, I stepped back and did what I should have done months earlier: checked what else has been published along this line since Apollo Research's [February 2025 paper](https://arxiv.org/abs/2502.03407). These are my initial reads — I haven't fully absorbed every paper — and the goal here is to map the landscape and flag which parts of my work have live neighbors in the literature.

## Live scoop risks

Three papers sit closest to what I've been doing.

**Nordby, Pais, Parrack — "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling"** ([arXiv 2604.13386](https://arxiv.org/abs/2604.13386), April 2026). Tests 12 models across Llama, Mistral, and **Qwen (0.5B, 3B, 14B, 72B)**. Runs a layer sweep, proposes multi-layer ensembling, characterises the deception feature as slowly rotating across depth. This is the biggest direct overlap with my Qwen replication and 32-layer sweep. Their headline result puts optimal probe layers in the **latter two-thirds of depth (mean ~65%)**, which directly contradicts my layers-10-15 finding at 31–47% depth on Qwen 3.5-4B. Worth investigating whether the disagreement is about (a) Qwen 3.5-4B specifically, which is a hybrid Gated DeltaNet + Attention architecture their sweep doesn't isolate, (b) the 4B scale versus their larger Qwen variants, or (c) differences in training data / probe method. Either way the disagreement is publishable rather than a problem.

**Wang et al. — "Beyond Prompt-Induced Lies"** ([arXiv 2508.06361](https://arxiv.org/abs/2508.06361), ICLR 2026 Oral). Evaluates 16 LLMs including multiple Qwen variants on a graph-reachability task and explicitly distinguishes **fabrication (commission-like) and concealment (omission-like)** as separate derivable scores. This is the closest prior art to my commission-vs-omission framing. The key difference: Wang et al. measure behavior, not activations — no probes, no layer analysis, no claim about mechanistic distinctness inside the model. The conceptual split is in the air and published in an oral-accepted venue, so I should frame my work as the mechanistic-activation complement to their behavioral result, not as first-mover on the split itself.

**Natarajan et al. — "Building Better Deception Probes Using Targeted Instruction Pairs"** ([arXiv 2602.01425](https://arxiv.org/abs/2602.01425), February 2026). Gemma-2-9B only. Builds a taxonomy of deception types (white lies, concealment, evasion) and trains targeted probes, reporting +20.5% AUC improvements over untargeted baselines. Closest to my "multiple probes for different deception types" direction. Their categories are taxonomic/behavioral (white lie / concealment / evasion) rather than mechanistic (commission / omission), and they report that layer choice explains only 2.7% of variance in their setup — a weaker role for layers than I'm finding on Qwen. They also don't connect to neuroscience.

## Partial overlaps worth citing

**Laurito et al. — "Liars' Bench"** ([arXiv 2511.16035](https://arxiv.org/abs/2511.16035), November 2025). 72,863 on-policy lie/honest examples from Llama-3.3-70B, **Qwen-2.5-72B**, Mistral-Small-24B, and Gemma-3-27B. Reports mean probes averaging balanced accuracy 0.54 — close to chance — and identifies that probes systematically fail on certain lie types. Documents the general cross-domain failure pattern I hit, though without the specific inverted-direction signature or the commission/omission mechanistic split.

**"Caught in the Act"** ([arXiv 2508.19505](https://arxiv.org/abs/2508.19505), September 2025). Probes Llama, **Qwen (1.5B–14B)**, and DeepSeek-r1. Runs an explicit layer-wise sweep and reports a "three-stage pattern": near-random early, peak mid, slight decline late. Qualitatively consistent with my commission-probe shape, though they don't carve by deception type.

**Smith, Chughtai, Nanda — "Difficulties with Evaluating a Deception Detector for AIs"** ([arXiv 2511.22662](https://arxiv.org/abs/2511.22662), November 2025). Methodological critique from Apollo/DeepMind-adjacent authors arguing cross-domain transfer is under-tested and labels are underdetermined. Friendly to my framing — this is the kind of paper my insider-trading result is a concrete instance of.

**Parrack, Hex, Nardo — "Trusted monitoring, but with deception probes"** ([LessWrong, July 2025](https://www.lesswrong.com/posts/eaEqAzGN3uJfpfGoc/trusted-monitoring-but-with-deception-probes)). Apollo-affiliated benchmarking of their two probes across five deception datasets, reporting 97% TPR at 1% FPR and showing one probe generalizes to backdoored code. Llama-3.3-70B only. No Qwen, no category split.

**Boxo et al. — "Benchmarking Deception Probes via Black-to-White Performance Boosts"** ([arXiv 2507.12691](https://arxiv.org/abs/2507.12691), January 2026). Compares probe-augmented white-box monitoring against black-box. Finds "weak but encouraging" gains. Evaluation-methodology focus, not categorization-focus.

**Fang et al. — "MM-DeceptionBench"** ([arXiv 2512.00349](https://arxiv.org/abs/2512.00349), December 2025). Multimodal, game-theory framing. Names **six deception categories explicitly, including deliberate omission and fabrication**. Behavioral benchmark, not probe-based. Reinforces that "omission as its own category" is a thread multiple groups are pulling on.

## Adjacent lines that don't directly overlap

**Anthropic — "Alignment Faking Revisited"** ([alignment.anthropic.com, 2025](https://alignment.anthropic.com/2025/alignment-faking-revisited/)). Improved classifiers on Llama-3 variants, Llama-4, and Qwen. Alignment-faking specific, not general deception categorization.

**Anthropic interpretability work on "strategic manipulation" vs "concealment" features** — reported via dictionary-learning methods in late 2025. Implicitly treats concealment as its own feature axis; no public paper carving commission vs omission as a formal hypothesis.

**Sandbagging detection** — multiple papers (Apart Research's "Sandbag Detection through Model Degradation"; [arXiv 2508.00943](https://arxiv.org/abs/2508.00943) "LLMs Can Covertly Sandbag"; Sullivan "Playing Dumb" 2026). All address sandbagging detection but via noise injection, CoT monitoring, or framing consistency — **none use dedicated activation probes trained on paired sandbag/non-sandbag data** the way my planned phase 2 does.

## Where my angle still looks open

Three pieces of the current project don't have a direct twin in what I've read so far:

1. **Neuroscience-guided probe categorization.** I found no LLM-deception paper citing Knappe 2025 or Lisofsky 2014 as a rationale for choosing probe categories. The neuro-AI survey literature goes the other direction (using LLMs as brain models). Using human deception neuroscience as a categorization framework for activation probes appears to be a gap.
2. **The specific inverted-AUROC-0.458 failure mode** on insider trading with a paired-facts (RepE) probe on Qwen. Liars' Bench documents broad probe failures; no one I found has published this exact inversion signature on this dataset.
3. **The combination** of (a) layer-profile comparison, (b) mechanistic commission/omission split, (c) neuroscience grounding. Wang et al. has the behavioral split; Natarajan et al. has a taxonomic probe family; nobody I found has fused all three.

## What this means for the project

Publish the phase 2 results before too much more time passes. The commission/omission distinction exists behaviorally in the literature as of ICLR 2026 and taxonomically as of February 2026, so the mechanistic-activation framing needs to land while it's still unclaimed. The Nordby et al. layer-depth disagreement is worth treating as a live question rather than a competitor — it's the kind of empirical tension that makes for a useful follow-up section.

I may come back and revise this post as I actually read each paper end-to-end. The summaries above are from an initial survey pass, not a close read.

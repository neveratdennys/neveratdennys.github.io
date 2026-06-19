---
title: "Two Probes, One Confusion Matrix"
published: true
summary: "A two probe approach for deception detection on Qwen 3.5-4B. Each probe handles its own training domain: commission for stating known false facts, omission for withholding insider information. Switching to the right probe for the scenario recovers +0.36 AUROC on insider trading over a single commission probe."
topics:
  - Interpretability
  - Multi probe
  - Qwen 3.5
---

**Recap.** This extends Apollo Research's [deception probe pipeline](https://arxiv.org/abs/2502.03407) on Qwen 3.5-4B.

**TL;DR**

- A two probe system recovers **+0.36 AUROC on insider trading**: commission [11, 14] scores 0.4581 there, a dedicated omission probe [17, 19] scores 0.8176. Each probe handles its own training domain. Not one probe that does both, but two probes that each cover their own lane.
- Layer picks matter per probe. Commission peaks at layers 11 to 14, omission at 17 to 19. Using one probe's layers on the other costs 0.10 to 0.17 AUROC.
- For commission, a trained combined LR probe works. For omission it collapses cross domain to near chance, while score averaging at inference holds around 0.71. The asymmetry is reproducible; the cause is untested.
- The omission probe is **not** a standalone detector. Its vs Alpaca score on its own training domain is 0.32, below chance. It discriminates honest from deceptive within a known scenario, but does not calibrate against arbitrary text.

The premise: a single commission probe, trained to catch stated falsehoods, inverts on insider trading, where the deception is concealment rather than a stated falsehood. If commission (stating known falsehoods) and omission (withholding known truths) do not share a single linear direction in the model's activations, they need separate probes ([background]({{ '/writing/deception-detection-two-signals/' | relative_url }})). This post builds that two probe system and evaluates its performance.

## Probe picks

The Apollo paper uses a single layer (22 of 80) on Llama-70B. Qwen 3.5-4B has 32 layers, depth fractions do not map cleanly across scales, and I am running two probes rather than one. I picked layers per probe, keeping the best configuration for each. The probe training pipeline and the per layer sweep behind these picks are covered in [post 2]({{ '/writing/deception-detection-two-signals/' | relative_url }}).

**Commission, by ensemble size (held out roleplaying):**

| N | Best combination | RP H/D | RP vs Alpaca | RP R@1% |
|:-:|---|:-:|:-:|:-:|
| 1 | [14] | 0.7934 | 0.9798 | 0.4823 |
| **2** | **[11, 14]** | **0.7955** | 0.9908 | 0.7516 |
| 3 | [0, 11, 14] | 0.7955 | 0.9908 | 0.7516 |
| 4 | [0, 11, 14, 15] | 0.7931 | 0.9929 | **0.8298** |
| 5 | [10, 11, 14, 15, 25] | 0.7938 | 0.9919 | 0.7889 |

For omission the search picks [17, 19] at N=2, and the N=1 through N=3 picks land within 0.005 AUROC of each other (around 0.71 on roleplaying), so the size choice is within noise. I kept [17, 19].

Two results from the search:

1. **Small N wins for both probes.** Commission peaks at N=2, omission is flat across N=1 to 3. Larger ensembles do not help. Apollo's single layer default is the right shape, just at a different layer than they used.
2. **The probes peak at different depths.** Commission at 11 to 14, omission at 17 to 19. Cross using the picks is a real loss: commission's layers on omission score 0.586 (barely above chance), omission's layers on commission score 0.698 (about 0.10 below commission's own). One global layer choice cannot be optimal for both.

## Probe methods

Score averaging is an approximation to a trained combined LR probe, so I checked both.

Commission [11, 14] trained as a combined LR matches its score averaging prediction within tolerance (roleplaying H/D 0.7907 against 0.7955, vs Alpaca 0.9897 against 0.9908). Commission deploys as a trained combined LR.

Omission is different. The trained combined LR collapses cross domain to near chance, worse than any of its own single layer parts, and a regularization sweep barely moves it. Score averaging at inference, applying each layer's probe separately and averaging the scores, holds around 0.71 and wins. So omission deploys as a score averaging ensemble over [17, 19]. Why this asymmetry appears on omission and not commission is unexplained and left for follow up.

## Recovery on insider trading

The commission probe inverts on insider trading at AUROC 0.4581, below chance. A dedicated omission probe recovers it to 0.8176, a gain of +0.36.

| Probe | Layers | Method | IT val AUROC |
|---|:-:|:-:|:-:|
| Commission baseline | [10, 11, 13, 14, 15] | Trained combined LR | **0.4581** |
| Commission updated | [11, 14] | Trained combined LR | 0.4590 |
| **Omission** | **[17, 19]** | **Score averaging** | **0.8176** |
| **Recovery** | | | **+0.36** |

This is per domain specialization, not a single probe that generalizes. The omission probe wins because it is evaluated on its own training domain. Both commission picks invert on insider trading because it is the wrong category for that probe.

## Cross probe comparison

Both probes across the shared evaluation suite (from `results/06_multi_probe_poc/confusion_matrix.csv`):

| Dataset | Probe | Relation | n honest | n deceptive | AUROC H/D | AUROC vs Alpaca | Recall at 1% FPR |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| RepE val | Commission [11, 14] | in domain | 61 | 61 | **1.0000** | 0.999 | 0.820 |
| Insider Trading val | Commission baseline | cross domain | 4,848 | 1,185 | **0.4581** | 0.978 | 0.000 |
| Insider Trading val | Omission [17, 19] | in domain | 949 | 236 | **0.8176** | 0.316 | 0.004 |
| Roleplaying | Commission [11, 14] | cross domain | 370 | 1,099 | 0.7955 | 0.991 | 0.752 |
| Roleplaying | Omission [17, 19] | cross domain | 370 | 1,099 | 0.7083 | 0.822 | 0.242 |

- Each probe performs well on its own training domain (commission 1.000 on RepE, omission 0.8176 on insider trading), and each falls off on the other's domain.
- Both transfer to roleplaying, commission higher (0.7955 against 0.7083), because roleplaying lies are more commission like (the model says something false rather than withholds something true). Roleplaying is cross domain for both probes, so this is not a like for like comparison, and each probe still scores highest in its own domain (omission 0.8176 on insider trading against 0.7083 here).
- The omission cell on RepE val is absent, since RepE val has no Qwen rollouts to score against. It remains a later item.

The vs Alpaca column shows omission's one sided limit: it sits at 0.316 on its own domain against commission's roughly 0.99, so it discriminates within a known scenario but does not work as a standalone detector against arbitrary text.

## Findings and limitations

A commission probe scoring below chance (0.458) on a concealment task is recovered to 0.82 by a category matched omission probe with its own layer selection, with the category split taken from parallels in deception neuroscience rather than chosen after the fact. The probes place the two types along different directions and layers in activation space. That is consistent with distinct underlying mechanisms but is not direct proof of one. The method itself, a separate probe per category, is a common pattern (mixture of experts and similar), and the cross domain failure it addresses has been reported before. 

I cover the related probing work, including the closest prior result, in the [related work post]({{ '/writing/deception-detection-related-work/' | relative_url }}); the short version is that others have shown the failure and trained per type probes, but combining or routing them is still open. So the claim is limited to the case study and the per category layer analysis, not an entirely novel method and not an automatic routing system ready for deployment.

Limitations and next steps:

- **No router.** Recovery assumes the scenario category is known. A learned router that selects the probe is required for any real use, and I have not built one.
- **Omission training data.** The omission probe trains on insider trading rollouts, where deception is mixed with topic and scenario. The clean version is RepE style paired data: identical context, one prompt that shares all information and one that withholds a specific fact. That is the next dataset.
- **Sandbagging.** A third category, deliberate underperformance, has an existing benchmark. Adding it would test whether per category specialization extends past two.
- **Scope.** The layer picks are specific to Qwen 3.5-4B at this scale. Whether the split holds on other models or architectures is open.

Artifacts: probe weights and the cross probe comparison at `results/06_multi_probe_poc/`, the sweep at `results/05_omission_32layer_sweep/scores_by_layer.json`, search output at `results/03_repe_32layer_sweep/ensemble_size_search.json`. Probe loading in `deception_detection/multi_probe.py`.

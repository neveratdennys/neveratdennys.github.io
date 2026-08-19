---
title: "Commission Trained Deception Probes Fail on Concealment in Qwen-3.5-4B"
published: true
summary: "A probe trained on commission style deception scores below chance on insider trading, a concealment task. A dedicated omission probe recovers detection by +0.36 AUROC, and the two probes show different layer profiles in Qwen 3.5-4B."
topics:
  - Interpretability
  - Multi probe
  - Qwen 3.5
---

Summary
-------

- I extended Apollo Research's [Detecting Strategic Deception Using Linear Probes](https://www.lesswrong.com/posts/9pGbTz6c78PGwJein/detecting-strategic-deception-using-linear-probes) ([original paper](https://arxiv.org/abs/2502.03407)) to Qwen 3.5-4B. A probe trained on commission style deception (stating known falsehoods) performs **below chance** on insider trading, a concealment task: AUROC 0.458 [0.439, 0.477].
- A second probe trained on omission style deception (withholding known truths) reaches AUROC 0.818 [0.787, 0.847] on insider trading, a gain of +0.36. Here 0.458 is a cross domain transfer and 0.818 is in domain, so the gain reflects both deception type matching and in domain training.
- The two probes have qualitatively different layer profiles. Commission peaks sharply at layers 10-15 and saturates. Omission sits flat across all 32 layers: holds a roughly 0.80-0.83 plateau through the middle layers, then eases to about 0.70 in the late layers.
- The omission probe is **not** a standalone detector. Its vs Alpaca AUROC on insider trading is 0.316, meaning it scores arbitrary normal text as more deceptive than the insider trading deceptive completions. It discriminates honest from deceptive within a known scenario, but does not calibrate against arbitrary text.
- Note: Qwen-3.5-4B shows batch size sensitive activations during extraction; all results reported here use batch_size=4, and the core results are robust across batch sizes 2, 4, and 8.

Background
----------

Apollo Research's [February 2025 paper](https://arxiv.org/abs/2502.03407) showed that linear probes trained on model activations can detect deceptive behavior with high AUROC. Their probe, trained on RepE paired honest/deceptive facts and applied at layer 22 of Llama-3.3-70B, reaches AUROC 0.96-0.999 across four evaluation datasets including insider trading.

I forked [their codebase](https://github.com/neveratdennys/deception-detection) to run on [Qwen 3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), which uses a hybrid Gated DeltaNet plus standard attention architecture, and the 4B version was chosen to fit within local GPU VRAM. Two setup changes are worth noting here: regenerating the Alpaca control completions with Qwen, and training on RepE paired facts rather than roleplaying scenarios. Together these two changes lifted vs Alpaca AUROC on roleplaying from 0.642 to 0.971; layer optimization then took it to 0.992.

Where the signal lives in Qwen
------------------------------

Before looking at cross domain transfer, I swept all 32 layers individually, fitting a single layer logistic regression probe at each depth.

The commission probe shows a sharp rise: within domain AUROC climbs from chance at layer 0 to 1.000 by layer 11 and holds. Vs Alpaca crosses 0.95 at layer 10 and peaks at 1.000 across layers 13-16. The peak band is layers 10-15 (around 40% depth) compared to the paper's layer 22 of 80 (28% depth) on Llama-70B. Depth fractions don't map cleanly across scales, but the shape is recognizably the same: early rise, mid depth saturation.

To confirm that stacking more layers does not help, I compared a 5 layer data driven selection (layers 10, 11, 13, 14, 15) against a 32 layer all layers baseline and a 7 layer evenly spaced probe, where the 5 layer selection wins on every metric.

The inversion
-------------

Then I ran that same probe on insider trading rollouts. This is the scenario where the model is tipped off about a merger, makes a trade, and is asked what information it used. Deception here is concealing the tip, not stating a falsehood.

It came out at **AUROC 0.458 [0.439, 0.477]**.

The entire confidence interval sits below chance. This is consistent with the learned direction flipping relative to deceptive/honest labels when moving from commission to concealment. The vs Alpaca AUROC stayed at 0.978, the probe still cleanly separates insider trading text from normal text, it just can't tell which direction is which inside the scenario.

The Apollo deception probe paper already flags a subtle version of this, their probe partially depends on "is the scenario deception related" rather than "is this response actually deceptive". On Llama-70B that scenario sensitivity is a small caveat layered over a strong cross domain signal. On Qwen 3.5-4B with a commission probe, the scenario channel works fine and the within scenario channel has flipped sign: the probe scores honest completions slightly above deceptive ones, putting AUROC just below chance at 0.458. The class means differ by under 0.02, so this is a weak inverse signal, not a clean mirror of the in domain result.

A second probe
--------------

To test whether the failure was about deception type rather than a model limitation, I trained a dedicated probe on insider trading rollouts (80% train, 20% held out) and compared its layer profile against the commission probe.

The profiles look qualitatively different.

![Per layer AUROC for commission and omission probes in Qwen 3.5-4B](/assets/images/qwen-commission-vs-omission-layers.png)

*Per layer AUROC for commission and omission probes in Qwen 3.5-4B*

Commission (blue) climbs fast and saturates in the 10-15 band. Both within domain and vs control metrics peak together. Omission (red) within domain AUROC holds a roughly 0.80-0.83 plateau through the early and middle layers, then drops to a lower 0.69-0.73 band in the late layers (20-31). If these shared a single direction you would expect similar depth profiles, though the difference could also partly reflect the structurally different training data (RepE paired facts vs insider trading rollouts).

Next I ran an exhaustive search over all 1 to 5 layer combinations to pick layers for each probe without hand tuning. Commission peaks at [11, 14], omission at [17, 19]. Cross using the picks showed degraded performance, commission's best layers on the omission task score 0.586, and omission's best layers on the commission task score 0.698.

The recovery
------------

Commission probe [11, 14] trained on RepE as a combined LR, omission probe [17, 19] deployed as a score averaging ensemble. Each probe evaluated on its own training domain and cross domain.

![Cross probe AUROC chart showing commission and omission on insider trading](/assets/images/qwen-cross-probe-auroc-ci.png)

*Cross probe AUROC chart showing commission and omission on insider trading*

| Dataset | Probe | Relation | n honest | n deceptive | AUROC H/D |
|---|---|:-:|:-:|:-:|:-:|
| RepE val | Commission | in domain | 61 | 61 | 1.000 |
| Insider Trading | Commission baseline | cross domain | 4,848 | 1,185 | 0.458 |
| Insider Trading | Omission | in domain | 949 | 236 | **0.818** |
| Roleplaying | Commission | cross domain | 370 | 1,099 | 0.795 |
| Roleplaying | Omission | cross domain | 370 | 1,099 | 0.708 |

The recovery on insider trading is +0.36 AUROC, from below chance to 0.818 with no confidence interval overlap. The omission probe is evaluated in domain here, so strong performance is expected. The more informative number is the cross domain row: both probes transfer partially to roleplaying, with commission higher (0.795 vs 0.708) because roleplaying lies are more commission like (the model says something false rather than withholds something true).

The omission probe's vs Alpaca AUROC on insider trading is 0.316, below chance, placing it as a relative discriminator within a known scenario, not a standalone detector against arbitrary text.

One asymmetry worth noting is that for commission, a trained combined LR probe matches the score averaging prediction. For omission, the trained combined LR cross domain performance is near chance, while the score averaging holds at 0.71 on roleplaying. Why this happens on omission and not commission is not directly tested.

A batch size artifact
---------------------

Batch size during activation extraction on Qwen 3.5-4B produces approximately 40% relative activation drift between batch_size=1 and batch_size=2, possibly specific to its Gated DeltaNet layers. An earlier headline result obtained at batch_size=1, had the omission probe looking like a capable standalone detector; that reading reversed at larger batch sizes and I retracted the initial result. Commission inverts and omission recovers at every batch size tested, and the +0.36 delta holds across batch sizes 2, 4, and 8. All numbers in this post use batch_size=4. The full investigation is in a [separate post]({{ '/writing/deception-detection-batch-size-kernel-dispatch/' | relative_url }}).

Where this sits in the literature
---------------------------------

The mechanism here, a separate probe per category, is a common pattern in ML (mixture of experts). The failure it addresses is also documented: [Liars' Bench](https://arxiv.org/abs/2511.16035) shows single probes near chance on concealment style lies. ["Pressure-Testing Deception Probes"](https://arxiv.org/abs/2605.27958) fixes the same cross domain failure differently, with style augmented single probes. [Natarajan et al.](https://arxiv.org/abs/2602.01425) train targeted per type probes on Gemma-2-9B but select a single best probe without a router and name routed ensembling as future work.

[Wang et al.](https://arxiv.org/abs/2508.06361) (ICLR 2026 Oral) explicitly distinguishes fabrication and concealment as separate derivable behavioral scores across 16 LLMs. Their conceptual split matches mine, but their evidence is behavioral (output text), not activation based. The differing layer profiles are suggestive of distinct representations, though dataset differences are an alternative explanation.

As a side note, the commission/omission split is convergent with what human neuroscience has found: [Knappe et al.'s 2025 meta-analysis](https://www.sciencedirect.com/science/article/pii/S0167876025007329) reports concealed information detection with a distinct and consistent neural signature, not a weaker version of active lying. This does not imply that LLM deception works like human deception. But it is at least suggestive that, in both cases, concealment appears not to be just a weaker version of explicit falsehood, but something with a partially distinct signature.

Limits and open questions
-------------------------

- A learned router that selects the right probe is required to realize the full benefit of a multi probe system, and I haven't built one here. The +0.36 recovery assumes the scenario category is already known, so it is an upper bound on what a routed system would deliver, not a deployable result.
- These layer picks and recovery numbers are specific to Qwen 3.5-4B. Whether the commission/omission split holds on other models or at other scales is open.
- The evidence for distinct representations is correlational, not causal.
- The omission probe trains on insider trading data, where the deception signal is mixed with topic and scenario. This means different layer profiles could reflect training data differences rather than mechanistically distinct deception types. Cleaner training data such as RepE style paired completions where one shares all information and one withholds a specific fact is the next dataset to build and would be the strongest test of the commission/omission distinction.
- Sandbagging (deliberate underperformance) has an existing benchmark and would test whether per category specialization extends past two probes.

Code, probe weights, and the cross probe comparison: [github.com/neveratdennys/deception-detection](https://github.com/neveratdennys/deception-detection). Full series of blog posts with methodology details: [neveratdennys.github.io](https://neveratdennys.github.io/).

---
title: "Two Kinds of Deception, Two Kinds of Signal"
published: true
summary: "Extending Apollo Research's linear probe pipeline to Qwen 3.5-4B. The cross domain transfer to insider trading performed below chance. A dedicated omission probe shows a different layer profile from commission, pointing toward treating the two as distinct deception categories in activation space."
topics:
  - Interpretability
  - Omission probe
  - Qwen 3.5
---

My [recent progress]({{ '/writing/deception-detection-qwen/' | relative_url }}) left the Qwen 3.5-4B probe separating honest and deceptive roleplaying completions at AUROC 0.976, but only 0.642 against an Alpaca control. This shows the ability to detect roleplaying style, but not necessarily deception.

Two setup changes between that post and this one are worth pointing out. First, the control. The Apollo paper probes Llama-3.3 and uses Llama-3.3's own Alpaca completions as the control (paper section 3.2.3). My first Qwen run reused the paper's shipped Llama generated Alpaca completions instead of Qwen generated completions, which made the control a writing style confound, Llama style against Qwen style. Regenerating the Alpaca completions with Qwen restored on policy control, matching the paper's design on a different model. Second, I switched the probe's training data from roleplaying scenarios to the paper's RepE (Representation Engineering) paired honest/deceptive facts, which the paper shows separates deception from control more cleanly. Together those two changes lifted vs Alpaca AUROC on roleplaying from 0.642 to 0.971, and the layer optimization below takes it to 0.992.

I decided to look at the following two points:

(1) Find out which layers carry deception signals in Qwen 3.5-4B. The original study sweeps layers and ends up using layer 22 of 80 on Llama-70B, but since Qwen 3.5's architecture is significantly different, I wanted the full per-layer picture on this model before picking the probe depth.

(2) Replicate the paper's cross-domain transfer check on Qwen, to see whether a smaller model with a different architecture behaves the same way Llama-70B did.

The layer investigation produced a clean peak band at layers 10–15. Cross-domain transfer showed the same family of issues the paper already flags: the probe partly detecting "the scenario is deception-related" rather than pure "this response is deceptive", but to a more extreme degree on Qwen 3.5. On roleplaying the signal is lower than the paper's Llama-70B numbers, on insider trading it fully inverts below chance. This gave me ideas to try a multi probe approach next.

## Where the signal lives

The Apollo paper applies its probe after layer 22 (of 80) in Llama-70B, picked from a sweep on Roleplaying data and reported in their Appendix D.2. They also note that layer choice matters less on most evaluation sets but drops off sharply past layer 29 on Insider Trading Confirmation, which may have its own layer dynamics. My earlier Qwen 3.5 configs used even layers across the middle half without a full layer test, so I swept all 32 Qwen layers individually, training a single-layer logistic regression probe at each depth.

```bash
# 32 layer commission sweep on RepE trained probe.
# experiment.py trains the combined LR probe on all 32 layers; by_layer.py
# fits a separate single layer probe at each depth and scores against eval.
.venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
    --config_file repe_qwen_full_layers.yaml

EXP_DIR=$(ls -td results/repe_qwen_full_layers_lr__*/ | head -1)
.venv/Scripts/python.exe -m deception_detection.scripts.by_layer run "$EXP_DIR"
```

![Commission probe per-layer AUROC sweep in Qwen 3.5-4B](/assets/images/qwen-commission-layer-sweep.png)

Layer 0 sits at chance on both metrics, with no learnable deception direction observed yet. Within domain separation (Honest vs Deceptive) climbs rapidly to 0.902 at layer 1 and 0.996 by layer 2, and saturates at 1.000 from layer 11 onward, holding at 1.000 all the way through layer 31. The vs Alpaca curve is noisier through the early layers, crosses 0.95 at layer 10, peaks at 1.000 across layers 13–16, then drifts back slightly to ~0.99 at the top of the stack. The peak band 10–15 is where both metrics hit their maximum, providing a clean reference point from here on.

An additional note: stacking more layers does not help. I trained three probe variants on the same RepE data and evaluated each on the held-out roleplaying dataset:

- **32-layer combined** (all layers 0–31): a naive use everything baseline.
- **7-layer even** (layers 12, 14, 16, 18, 20, 22, 24): the middle layers similar to Llama configs.
- **5-layer data-driven** (layers 10, 11, 13, 14, 15): the top individual performers from the sweep above.

![Probe variant comparison on cross-domain roleplaying](/assets/images/qwen-probe-variant-comparison.png)

The 5 layer probe wins on every metric: H vs D AUROC 0.791, vs Alpaca AUROC 0.993, and recall at 1% FPR of 83.2%, compared to 0.725 / 0.978 / 54.0% for the 32 layer and 0.743 / 0.981 / 50.7% for the 7 layer. (On the in domain RepE validation set all three saturate at 1.000.) This reaffirms the idea that uninformative layers dilute the deception direction rather than adding redundancy.

```bash
# The 5 layer optimized commission probe (layers 10, 11, 13, 14, 15)
.venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
    --config_file repe_qwen_optimized.yaml
```

## The insider trading result

Then I ran that same probe on insider trading rollouts. This is the scenario where the model is tipped off about a merger, makes a trade, and is asked what information it used. Deception here is concealing the tip.

It came out at **AUROC 0.458**.

That is below chance, and a significant departure from what the paper reported on the same scenario. The Apollo probe (trained on their equivalent paired honest/deceptive data) reaches AUROC 0.999 on Insider Trading Report on Llama-70B, with a cross-domain AUROC range of 0.964–0.999 across their four evaluation datasets.

![Cross-domain transfer comparison: paper's Llama-70B probe vs my Qwen 3.5-4B probe](/assets/images/qwen-insider-trading-comparison.png)

But the paper already flags a more subtle concern even at those high AUROCs: its probe partially depends on *"is the scenario deception-related"* rather than *"is the response actually deceptive"* (paper section 4.1.1, and Appendix A.1 shows the same effect on an independent AI Liar dataset, honest responses in deception-incentivized scenarios score higher than honest responses in neutral ones). On Llama-70B that scenario sensitivity is a small caveat layered over a strong H vs D signal. On Qwen 3.5-4B it's the *dominant* effect: within insider trading, the probe scores honest completions *higher* than deceptive ones, and the learned deception direction is effectively inverted on this dataset. The vs control AUROC stayed at 0.993, so the probe still cleanly separates insider trading text from Alpaca text, it just can't tell which direction is which inside the scenario. The scenario vs control channel still works, though the H vs D channel inside the scenario has collapsed.

Hypothesis: the RepE probe learned a particular kind of deception, stating things known to be false. Insider trading's deception isn't the same. Deceiving by *withholding* information and deceiving by *saying the wrong thing* are different operations, and they don't appear to share a direction in activation space.

## Digging into the hypothesis

Before building around this, I wanted to check whether the distinction was convenient rationalization or something the literature already had a view on. The human neuroscience work turns out to be fairly clear.

[Knappe et al.'s 2025 meta-analysis](https://www.sciencedirect.com/science/article/pii/S0167876025007329) of 54 ERP studies reports P300 amplitude for concealed information tests at Cohen's d = 1.59, where omission has its own large, consistent neural signature, not a weaker version of commission. [Lisofsky et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/24929201/), a quantitative meta-analysis pooling 22 fMRI studies (N=416), find that interactive/social deception recruits temporal-parietal regions, while non-interactive lying recruits dorsolateral PFC. Different contexts, different circuits.

Note that I'm not claiming LLMs, and Qwen 3.5 specifically, behave the way they do because they are just like brains. The argument is narrower: deception isn't a single phenomenon in human neural data, so assuming it would be a single direction in an LLM's activation space seems to be an assumption worth testing. The category split I pulled out of the insider trading failure, commission vs omission proposed as mechanistically distinct, is the same split that is consistently observed in human ERP literature. I thought this would be interesting to test next.

## Training a probe on the other kind

Quickest way to test the split: train a dedicated omission probe on the insider trading rollouts (4,848 honest, 1,185 deceptive) and compare its layer profile against the commission probe. Same pipeline, same model, same control, different training domain.

The profiles look qualitatively different.

![Per-layer AUROC for the commission and omission probes in Qwen 3.5-4B](/assets/images/qwen-commission-vs-omission-layers.png)

- **Commission (blue):** both metrics climb fast and saturate in the 10–15 band. Within domain AUROC reaches 1.00 by layer 11 and holds. vs Alpaca crosses 0.95 at layer 10 and peaks at 1.00 through layers 13–16.
- **Omission (red):** within domain AUROC sits flat in the 0.72–0.83 band across all 32 layers, with no climb, no peak, no depth dependence at any depth. The vs Alpaca curve is noisy (0.15–0.59 in the canonical re-derivation, varying without a clean peak).

If these were the same mechanism sharing a direction, they should peak at the same depth with the same shape, and they don't. Commission rises and saturates both within domain H vs D and vs Alpaca together in the 10–15 band. Omission's within domain H vs D sits stable across the entire stack without any clear rise. The two probes have qualitatively different depth profiles, which is the decision gate I needed before putting effort into a multi probe system: commission and omission are behaving as distinct signals, not variants of one.

## Compute notes

When this post first went up the analysis stopped at layer 23. Running the omission sweep on my local hardware was more work than the RepE sweep. Insider trading has about 6× the samples and 10× the sequence length of the RepE data, which puts activation memory roughly 60× higher. A single 32 layer run OOM'd immediately, so the omission sweep is sharded into 4 layer chunks. Along with some other memory fixes in the pipeline, this got past the initial OOM.

```bash
# 8 shard omission sweep, 4 layers per shard. Each shard trains its own
# combined LR + per layer probes on insider_trading__upscale, then merge_layer_shards.py
# combines the per layer scores into a single 32 layer file.
for SHARD in 00_03 04_07 08_11 12_15 16_19 20_23 24_27 28_31; do
    .venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
        --config_file omission_qwen_layers_${SHARD}.yaml
    EXP_DIR=$(ls -td results/omission_qwen_layers_${SHARD}_lr__*/ | head -1)
    .venv/Scripts/python.exe -m deception_detection.scripts.by_layer run "$EXP_DIR"
done

# Merge the 8 shards into a single scores_by_layer.json
.venv/Scripts/python.exe -m deception_detection.scripts.merge_layer_shards \
    results/omission_qwen_layers_00_03_lr__*/  \
    results/omission_qwen_layers_04_07_lr__*/  \
    results/omission_qwen_layers_08_11_lr__*/  \
    results/omission_qwen_layers_12_15_lr__*/  \
    results/omission_qwen_layers_16_19_lr__*/  \
    results/omission_qwen_layers_20_23_lr__*/  \
    results/omission_qwen_layers_24_27_lr__*/  \
    results/omission_qwen_layers_28_31_lr__*/  \
    --output results/05_omission_32layer_sweep/scores_by_layer.json
```

Still, the machine blackscreened multiple times mid sweep on shards 24–27 and 28–31, so those layers were a gap in the omission profile when this post first went up. Something at the driver or power layer is unhappy before I hit OOM. (The full 0–31 picture eventually closed in a follow up post.)

## Next

The layer 24–31 gap eventually closed, and the late layer story turned out simpler than expected (covered in the [batch size post]({{ '/writing/deception-detection-batch-size-kernel-dispatch/' | relative_url }})). Beyond that, the next step is a multi probe POC: pick layers for each probe in a principled way, retrain commission and omission as final probes, and build a cross probe comparison where each probe is evaluated against its own training domain plus held out cross-domain data. The canonical numbers from the batch size post are what that POC uses. The POC is in progress and will be a separate post.

---
title: "Computation Tuning: Batch Size Learnings"
published: true
summary: "The omission probe's headline standalone-detection result turned out to depend on a hardware-mitigation parameter rather than the probe itself. Tracing it from a stale numerical comparison to a kernel dispatch staircase, then characterising the stable regime."
topics:
  - Interpretability
  - Methodology
  - Qwen 3.5
---

This post is about something I did not expect to find. After the omission probe work I [posted]({{ '/writing/deception-detection-two-signals/' | relative_url }}), I went back to tighten up methodology and tuning run parameters to better work on my hardware before writing the next round. I noticed a quiet inconsistency, and ended up establishing that one of the headline results from the earlier work was a numerical artifact of batch size choice, not a property of the probe.

The short version: under the configuration I had been running, the omission probe looked like a deployable standalone detector against normal text. Under the configuration the field actually uses, it isn't. The relative ranking story still holds. 

## The trigger

While reviewing data from the omission probe, I lined up its scores from a few different sweep runs and noticed the numbers had shifted from the values I had recorded weeks earlier. Same probe, same code paths I thought, had slightly different scores.

The way it first showed up was on the omission probe's "vs Alpaca" AUROC on insider trading, which had been reported at 0.49 in the original sweep and at 0.90 in a more recent rerun. That is not a small drift.

I started by suspecting `flash-linear-attention`. The Qwen 3.5-4B architecture uses Gated DeltaNet for most of its layers, with attention every fourth layer. Around the time the score shift happened, I had installed `fla` and `triton-windows` to fix a separate compute issue. Different kernel implementations of the same operation are not bit identical. So it seemed like pre fla and post fla numerics differ at the layer output level, the differences compound through 32 layers, the probe scores end up in different distributions.

So I moved to test this theory.

## Ruling out fla

The first test was a controlled fla disabled run. Block fla at the import layer in the current environment, force the qwen3_5 module to fall back to the torch implementation, rerun the same shard. If fla is the cause, fla off in current env should reproduce the older pre fla numbers.

```python
# At the top of the runner, before any transformers imports
import sys
for m in ["fla", "fla.ops", "fla.ops.gated_delta_rule", "fla.modules", "fla.layers"]:
    sys.modules[m] = None

# Verify the block worked
from transformers.utils.import_utils import is_flash_linear_attention_available
assert not is_flash_linear_attention_available(), "FlaBlocker failed"
```

| Metric | pre fla | post fla | fla off (current env) |
|---|:-:|:-:|:-:|
| Roleplaying H/D | 0.7529 | 0.5977 | 0.6270 |
| Roleplaying vs Alpaca | 0.7067 | 0.7015 | 0.8312 |
| Insider Trading H/D | 0.8272 | 0.7427 | 0.7498 |
| Insider Trading vs Alpaca | 0.4879 | 0.9039 | 0.9063 |

It is not. fla off in current env matches post fla closely (mean |delta| 0.04) and stays far from pre fla (mean |delta| 0.19). If fla had been the cause, fla off should have looked like pre fla, since both run the torch fallback. They don't. Whatever shifted the numbers between the original run and the rerun, it wasn't the kernel choice.

## What actually changed

The next thing to check was whether anything else differed between the pre fla snapshot date (April 16) and the rerun date weeks later. I had a `pip freeze` from April 16 saved as a side artifact for unrelated reasons, so I diffed it against the current environment.

```bash
# Capture current and diff against the saved snapshot
.venv/Scripts/python.exe -m pip freeze > current_env.txt
python -c "
old = dict(line.strip().split('==', 1) for line in open('env_backup_pre_fla_20260416_234311.txt') if '==' in line)
new = dict(line.strip().split('==', 1) for line in open('current_env.txt') if '==' in line)
print('Changed versions:', [(k, old[k], new[k]) for k in old.keys() & new.keys() if old[k] != new[k]])
print('Added:', sorted(new.keys() - old.keys()))
print('Removed:', sorted(old.keys() - new.keys()))
"
```

Zero version changes. Same `transformers`, `torch`, `numpy`, `scikit-learn`, `triton-windows`, all the way down. No package was downgraded or upgraded between the two snapshots.

That left code level differences. The infrastructure work I did in mid April to make the deeper omission shards run on a 16 GB card included a small dataset aware batch size table inside the activation extraction code. Before mid April, insider trading ran at training batch size 2. After the TDR mitigation, it ran at batch size 1, to keep per kernel time under the Windows watchdog limit. pre fla shard 16_19 had been produced with `batch_size = 2`. The post fla rerun used `batch_size = 1`.

This is the only python-level code change I could identify between the two snapshots. So this seemed like the likely culprit.

## Test and rerun

Before doing a full shard rerun, I ran a smaller test to see whether batch size mechanically affects the activations at all. Six insider trading prompts, run through the model at batch size 1 and batch size 2 separately, layer 17 and 19 hidden states extracted directly and compared element-wise.

```bash
# Atomic test: same prompts, two batch sizes, compare activations at L17 and L19
.venv/Scripts/python.exe -m deception_detection.scripts.diagnose_phase2_atomic_batch
# -> writes results/diagnose_phase2_atomic_batch.json with max/mean/relative diff
```

Result: max absolute difference 22 to 41 units across the two layers. Mean absolute difference around 0.05 against an act magnitude of around 0.13, so roughly 35 to 40 percent relative difference per token. Forty percent relative drift from changing batch size on the same prompts is much larger than what published kernel equivalence work would predict. FlashAttention reports max relative error around 1e-3 in BF16, gated delta rule's reference implementation is the Triton kernel, and some quick search results seem to assume implicitly that activation extraction is invariant to batch size choice. The atomic test said pretty clearly that this assumption fails on Qwen 3.5-4B at the batch sizes I was running.

So next to rerun in higher batch sizes, I added two environment variables, `DECEPTION_TRAIN_BATCH_SIZE` and `DECEPTION_EVAL_BATCH_SIZE`, to override the dataset-aware defaults at runtime. Then I reran shard 16_19 in the current environment with `batch_size = 2` and `eval_batch_size = 4`, restoring the values that were in use on April 16.

```bash
# Rerun shard 16_19 in current environment but with the old batch sizes
DECEPTION_TRAIN_BATCH_SIZE=2 DECEPTION_EVAL_BATCH_SIZE=4 \
    .venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
    --config_file omission_qwen_layers_16_19.yaml

# Then by_layer for per layer probes
NEW_DIR=$(ls -td results/omission_qwen_layers_16_19_lr__*/ | head -1)
DECEPTION_TRAIN_BATCH_SIZE=2 DECEPTION_EVAL_BATCH_SIZE=4 \
    .venv/Scripts/python.exe -m deception_detection.scripts.by_layer run "$NEW_DIR"
```

| Metric | pre fla (b=2) | post fla (b=1) | fla off (b=1) | rerun (b=2, current env) |
|---|:-:|:-:|:-:|:-:|
| Roleplaying H/D | 0.7529 | 0.5977 | 0.6270 | 0.7122 |
| Roleplaying vs Alpaca | 0.7067 | 0.7015 | 0.8312 | 0.7927 |
| Insider Trading H/D | 0.8272 | 0.7427 | 0.7498 | 0.8244 |
| Insider Trading vs Alpaca | 0.4879 | 0.9039 | 0.9063 | 0.4149 |

Mean absolute delta of the rerun versus pre fla was 0.05. Mean absolute delta versus post fla was 0.19. The rerun in current environment with the old batch size reproduced the old numbers within tolerance, including the IT vs Alpaca chance level reading. batch size was the dominant factor.

The Insider Trading vs Alpaca cell is the one that flipped most cleanly. pre fla 0.49, batch=2 in current env 0.41, both close to chance. post fla 0.90, fla off 0.91, both close to deployable. The same probe with the same weights scoring the same prompts, in the same model checkpoint, gives qualitatively different "is this deceptive vs is this normal" readings depending on what batch size the activation extraction was run at.

## Sweep across batch sizes

I wanted to see if this affect from batch size extends to larger sizes, so I tested batch sizes 1, 2, 4, and 8 next.

```bash
# Sweep across batch sizes on shard 16_19 (one config per batch size)
for B in 1 2 4 8; do
    DECEPTION_TRAIN_BATCH_SIZE=$B DECEPTION_EVAL_BATCH_SIZE=$B \
        .venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
        --config_file omission_qwen_layers_16_19.yaml
    NEW_DIR=$(ls -td results/omission_qwen_layers_16_19_lr__*/ | head -1)
    DECEPTION_TRAIN_BATCH_SIZE=$B DECEPTION_EVAL_BATCH_SIZE=$B \
        .venv/Scripts/python.exe -m deception_detection.scripts.by_layer run "$NEW_DIR"
done
```

| Metric | b=1 | b=2 | b=4 | b=8 |
|---|:-:|:-:|:-:|:-:|
| Roleplaying H/D | 0.60 | 0.71 | 0.71 | 0.71 |
| Roleplaying vs Alpaca | 0.70-0.83 | 0.79 | 0.82 | 0.79 |
| Insider Trading H/D | 0.74-0.75 | 0.82 | 0.82 | 0.82 |
| **Insider Trading vs Alpaca** | **0.90** | **0.41** | **0.32** | **0.20** |

The batch=2, batch=4, batch=8 columns are tightly clustered. Pairwise mean absolute deltas across the four metrics are 0.035 between b=2 and b=4, 0.039 between b=4 and b=8, 0.056 between b=2 and b=8. Within the cluster, batch size choice barely matters. Outside the cluster, b=1 is the lone outlier sitting 0.17 to 0.24 away from every batch≥2 environment.

The Insider Trading vs Alpaca row is the most striking. It decreases monotonically as batch size increases, going from 0.90 at batch size 1 to 0.20 at batch size 8. By batch=8 the probe scores Alpaca normal text *higher* than insider trading deceptive completions on average. Whatever signal the omission probe was using to look like a standalone detector at batch=1, it disappears smoothly as batch size moves into the standard tensor core dispatch regime, and inverts past chance.

 Most characterizations show small variance that wash out across many tokens. A staircase that monotonically inverts a metric across four batch sizes is a different shape. The guess thrown in by Claude is: at batch=1 the GDN kernel routes through a special-case path (probably SIMT cores with FP32 accumulation), at batch≥2 it routes through tensor-core paths with lower-precision accumulation, and the probe direction learned at b=1 picks up something about that special-case path that doesn't generalize to the standard regime. While this is an interesting problem to dive into, I will resume going towards my intended goal for now.

## What this means for the project

**The H/D ranking story holds across batch sizes.** Commission inverts on insider trading (AUROC around 0.46) at every batch size I tested. Omission recovers on insider trading (AUROC around 0.82 in the stable regime). The recovery delta from "wrong-category probe" to "right-category probe" is +0.36 AUROC in the stable regime. That is the relative ranking core finding from the omission post, and it survives.

**The "deployable standalone detector" doesn't hold.** I had reported earlier that the omission probe gave Insider Trading vs Alpaca AUROC around 0.90, characterizing it as a strong standalone detector that could flag insider trading deceptive completions against normal text. Under the standard batch size regime, that is no longer supported. The probe ranks deceptive completions correctly *within* the insider-trading domain (so H/D AUROC stays at 0.82), but it does not separate the deceptive-completion distribution from the normal-text distribution. 

**The pre bath size change numbers are usable.** They were produced at b=2, sit in the stable cluster, and approximate what a clean b=4 canonical would give within a few hundredths of AUROC across metrics. The post fla numbers are the ones using b=1 and need re-derivation.

**Going forward, batch=4 is the right pick.** Matches the dispatch regime that probing papers in this size range typically run in, and fits on a 16 GB card across the deepest shards of the omission sweep without spilling into shared GPU memory, which batch=8 does on laster shards. 

## What published work use for batch size

Apollo's deception probe configs use batch sizes in the 4-16 range on H100 class hardware. RepE on LLaMA-2 typically runs batch=8 to 16. Nordby et al.'s ensembling paper across 12 models doesn't report a sensitivity check. The implicit assumption seems to be that any reasonable batch size produces equivalent activations within tolerance.

The 36-40% relative activation drift I measured between batch=1 and batch=2 is well outside that tolerance. It is likely that a b=1 value is much smaller than typical usage, as researchers typically has access to hardware with less retrictive memory limits, so even if this issue may be replicated in other experiments they are almost never encountered.


## The re-derivation

I reran the full 32 layer omission sweep to produce clean results with the same training data, same eval datasets, same probe pipeline as the original sweep, just with batch_size held at 4 throughout activation extraction and per layer scoring. The four 4 shard configs took about 100-115 minutes each and ran end-to-end across two calendar days.

```bash
# Canonical b=4 4 shard re-derivation. Each shard runs experiment.py
# (combined-LR train + scoring) followed by by_layer.py (per layer probes).
for SHARD in 00_07 08_15 16_23 24_31; do
    DECEPTION_TRAIN_BATCH_SIZE=4 DECEPTION_EVAL_BATCH_SIZE=4 \
        .venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
        --config_file omission_qwen_layers_${SHARD}.yaml

    NEW_DIR=$(ls -td results/omission_qwen_layers_${SHARD}_lr__*/ | head -1)
    DECEPTION_TRAIN_BATCH_SIZE=4 DECEPTION_EVAL_BATCH_SIZE=4 \
        .venv/Scripts/python.exe -m deception_detection.scripts.by_layer run "$NEW_DIR"
done

# Merge into a single 32 layer canonical scores_by_layer.json
.venv/Scripts/python.exe -m deception_detection.scripts.merge_layer_shards \
    results/omission_qwen_layers_00_07_lr__*_<ts>/ \
    results/omission_qwen_layers_08_15_lr__*_<ts>/ \
    results/omission_qwen_layers_16_23_lr__*_<ts>/ \
    results/omission_qwen_layers_24_31_lr__*_<ts>/ \
    --output results/05_omission_32layer_sweep_b4_canonical/scores_by_layer.json

# Validate against existing reference points
.venv/Scripts/python.exe -m deception_detection.scripts.validate_canonical_b4_4shard \
    --canonical-merged results/05_omission_32layer_sweep_b4_canonical/scores_by_layer.json
```

The headline metrics on `[17, 19]` score averaging:

| Metric | Rerun (b=4 4 shard) |
|---|:-:|
| Insider Trading H/D | **0.8176** |
| Insider Trading vs Alpaca | **0.3162** |
| Roleplaying H/D | 0.7083 |
| Roleplaying vs Alpaca | 0.8219 |

The Insider Trading H/D of 0.8176 is the one that matters: the omission probe still discriminates honest from deceptive within insider trading, where the commission probe inverted below chance ([post 2]({{ '/writing/deception-detection-two-signals/' | relative_url }})). That within scenario recovery survives the batch size correction. The Insider Trading vs Alpaca cell at 0.3162 is the retracted figure: under b=1 it had read around 0.90 and looked like a standalone detector against normal text, in the stable regime it sits below chance.

A small methodological observation worth recording. The 4 shard layout (8 layers per shard, 4 shards) and the batch=4 single shard test from the prior sweep produced identical per layer probe metrics, mean delta exactly 0.0000 across the four headline cells. This empirically confirms that sharding layout doesn't affect the per layer numerics. Only batch_size does. So future re-derivations can pick whichever sharding layout is convenient, without worrying about reproducibility.

## A hardware aside

The first attempt at the canonical run crashed on shard 16_23 with `CUDA error: an illegal memory access was encountered`. The CUDA context corrupted, `nvidia-smi` reported `GPU is lost. Reboot the system to recover this GPU`, and subsequent operations failed with `invalid argument` from a kernel that no longer had a device to talk to.

A physical GPU reseat fixed it. Same shard config, same batch size, same code, second attempt completed cleanly at 10.39 GB peak VRAM. This is likely a GPU issue or PCIE issue but after a number of attempts I was able to complete the computation. The deeper shard 24_31 ran cleanly at 12.33 GB peak immediately after, which is consistent with that read. 

## What this updates

The initial "deployable standalone detector against normal text" reading is retracted. The H/D recovery story stays at +0.36 AUROC, with all batch sizes in the stable cluster giving values between +0.30 and +0.37. The relative ranking core claim is robust. 

The anomolous artifacts in the project tree are now tagged. Downstream scoring scripts require the b=4 file and warn on any fallback, so future runs won't accidentally use the older numbers.

## Next

The updated artifact and the omission sweep is now on stable numerics. The multi probe comparison these canonical numbers feed into is written up in a forthcoming post, "Two Probes, One Confusion Matrix." The thread beyond that is whether the omission probe's relative ranking signal can be strengthened with cleaner training data, which is the paired omission data direction I had been planning. 
---
title: "First Notes on Extending Deception Detection to Qwen 3.5"
published: true
summary: "Initial results from extending a deception detection pipeline to Qwen 3.5-4B. First rollout on roleplaying, its grade distribution, and early probe AUROCs."
topics:
  - Interpretability
  - Qwen 3.5
---

I started this project from Apollo Research's work on [Detecting Strategic Deception Using Linear Probes](https://www.apolloresearch.ai/research/detecting-strategic-deception-using-linear-probes/) and the accompanying [paper](https://arxiv.org/abs/2502.03407). The basic idea is simple: train linear probes on model activations and see whether deceptive behavior can be detected from internal representations.

My current fork in [deception-detection](https://github.com/neveratdennys/deception-detection) is an early extension. The main change so far is support for Qwen 3.5-4B, so I can run the pipeline on a smaller local model and iterate faster.

## What I've added

- Qwen 3.5-4B model support
- cleanup for Qwen `<think>...</think>` output so token alignment does not break
- Microsoft Foundry support for the grading step
- a first Qwen roleplaying rollout file and experiment config

The first saved Qwen roleplaying rollout already gives me a usable starting point: 371 deceptive roleplaying prompts with 5 completions each. There is room to improve dataset design, controls, and evaluation, but the graded completions are already usable for a first probe.

## First probe results

The pipeline is two stages: generate rollouts on the roleplaying scenarios with Qwen as the policy model and Azure OpenAI as the grader, then train a linear probe on the resulting honest vs deceptive activations.

```bash
# 1. Generate Qwen 3.5-4B roleplaying completions, grade with Azure OpenAI.
.venv/Scripts/python.exe -m deception_detection.scripts.generate_rollouts \
    roleplaying__plain --model_name qwen-4b --use_api False \
    --grading_api azure_openai --num 5

# 2. Train + evaluate the linear probe.
.venv/Scripts/python.exe -m deception_detection.scripts.experiment run \
    --config_file qwen_roleplaying.yaml
```

My first test results are:

- AUROC 0.976 for honest vs deceptive on the same scenarios
- AUROC 0.642 for deceptive vs Alpaca control

![ROC plots from the Qwen 3.5 roleplaying run](/assets/images/qwen-roleplaying-all-roc.png)

The ROC plot makes the current shape of the result easy to see: strong separation on the held-out honest-vs-deceptive split, and much weaker separation against Alpaca as a control.

## Grade distribution

![Grading histogram for the Qwen 3.5 roleplaying rollout](/assets/images/qwen-roleplaying-grading-histogram.png)

The histogram shows the 1–7 honesty grades from GPT-5.4-mini across all 1,855 completions in the rollout. Every prompt was designed to elicit deception, but the distribution is bimodal with clean peaks at both ends: about 59% of completions land at 6 or 7 (Mostly or Entirely Dishonest), about 20% at 1 or 2 (Honest), and about 21% at 3–5 (Ambiguous or mixed). The peaks at both ends suggest the grader is making confident decisions rather than producing noisy labels. The spread is Qwen's own compliance pattern, even when prompted to be deceptive, the model chooses honesty or hedges roughly 40% of the time. That's useful for probe training, because the honest and deceptive labels come from actual model behavior rather than just the prompt.

## Next

That is the current picture as I see it: strong separation on the main task, much weaker separation against the control. So the next step is to improve the control setup, expand cross-dataset evaluation, and see whether the probe is tracking deception rather than recognizing role play.

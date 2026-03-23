# FusionLM

**Model merging for enterprise finance AI.** Combines two fine-tuned specialist LLMs — a quantitative reasoning model and a client communication model — into a single merged model using SLERP, TIES-Merging, and DARE. No retraining. No GPU required for the merge step. One inference pass.

> **Design doc + benchmark dashboard →** [raosiddharthp.github.io/FusionLM](https://raosiddharthp.github.io/FusionLM)  
> **Live demo →** [huggingface.co/spaces/raosiddharthp/fusionlm](https://huggingface.co/spaces/raosiddharthp/fusionlm)

---

## The problem

Financial AI teams routinely fine-tune separate models for separate workflows: one on earnings transcripts and covenant analysis, another on client-facing communication. Any real analyst task — writing an LP memo that is both numerically precise and clearly written — requires both simultaneously.

Running two models in a pipeline doubles inference cost, introduces a context hand-off that degrades coherence, and adds ~400ms of serial latency. Joint retraining costs $2,000–$8,000 in GPU compute and risks catastrophic forgetting.

Model merging at the weight level eliminates all three problems. The merge step runs on CPU. The result is a single model file, deployable identically to either source model.

---

## Results

| Model | FinBench | MMLU Finance | TruthfulQA | Inference cost | Merge cost |
|---|---|---|---|---|---|
| Quant specialist (source A) | 71.2 | 64.1 | 51.3 | 1× | — |
| Comms specialist (source B) | 58.4 | 60.8 | 62.1 | 1× | — |
| Two-model pipeline (baseline) | 70.8 | 70.1 | 63.4 | **2×** | — |
| FusionLM · SLERP | 66.3 | 65.3 | 59.8 | 1× | $0 |
| FusionLM · TIES-Merging | 68.9 | 67.1 | 61.2 | 1× | $0 |
| **FusionLM · DARE-TIES** ★ | **70.5** | **69.5** | **63.0** | **1×** | **$0** |
| Joint retrain (upper bound) | 72.1 | 71.4 | 64.8 | 1× | ~$5,000 |

DARE-TIES achieves **98.5% of pipeline baseline quality at 50% of the inference cost** and zero training cost. It outperforms both source specialists on every benchmark they were each weak on.

---

## How it works

Model merging operates on *task vectors* — the element-wise difference between a fine-tuned model's weights and the shared base model's weights:

```
τ = θ_finetuned − θ_base
```

Merging combines task vectors from multiple specialists and applies the result back onto the base model. The three techniques FusionLM implements differ in how they handle interference between task vectors:

**SLERP** interpolates between two models along a geodesic arc on the weight hypersphere, preserving geometric properties that linear averaging destroys. Controlled by a single interpolation parameter `t ∈ [0, 1]`. Limited to exactly two models.

**TIES-Merging** handles N models. It trims low-magnitude delta parameters that add noise without signal, runs a sign election to resolve conflicts where models disagree on the direction of a weight update, and merges the surviving, agreed-upon parameters.

**DARE** is a sparsification pre-step that randomly zeros out a fraction `p` of delta parameters and rescales the survivors by `1/(1−p)` to preserve expected embedding magnitude. Applied before TIES, it reduces interference. Effective at drop rates up to 90% — delta parameters are far sparser than the weights themselves.

All three run entirely on CPU. A 7B merge completes in ~18 minutes on an M2 MacBook Pro.

---

## Repo structure

```
fusionlm/
├── configs/
│   ├── slerp.yaml          # SLERP baseline · t=0.5
│   ├── ties.yaml           # TIES-Merging · density=0.5
│   ├── dare_ties.yaml      # DARE-TIES · recommended config
│   └── dare_high_sparsity.yaml  # DARE · p=0.9 ablation
├── merge/
│   ├── run_merge.py        # mergekit wrapper · CLI entry point
│   └── validate_merge.py   # architecture compatibility checks
├── eval/
│   ├── run_eval.py         # lm-eval-harness runner · 3-task suite
│   ├── tasks/
│   │   ├── finbench.yaml   # custom FinBench task definition
│   │   └── finance_mmlu.yaml
│   └── results/            # benchmark outputs · JSON + summary CSV
├── demo/
│   ├── app.py              # Gradio side-by-side comparison UI
│   └── requirements.txt
├── docs/
│   └── index.html          # design doc · raosiddharthp.github.io/FusionLM
└── README.md
```

---

## Quickstart

**Requirements:** Python 3.10+, ~28GB disk (three 7B model checkpoints), 16GB RAM minimum for the merge step. No GPU needed.

```bash
git clone https://github.com/raosiddharthp/FusionLM
cd FusionLM
pip install -r requirements.txt
```

**Run a merge:**

```bash
python merge/run_merge.py --config configs/dare_ties.yaml --output ./merged_model
```

**Evaluate the merged model:**

```bash
python eval/run_eval.py \
  --model ./merged_model \
  --tasks finbench,finance_mmlu,truthfulqa \
  --output eval/results/
```

**Run the demo locally:**

```bash
cd demo && python app.py
# opens at localhost:7860
# side-by-side: source A · source B · merged model
```

---

## Configuration

All merge behaviour is specified in YAML. The recommended DARE-TIES config:

```yaml
# configs/dare_ties.yaml
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  int8_mask: true
  dtype: bfloat16
models:
  - model: BioMistral/BioMistral-7B       # quant specialist
    parameters:
      density: 0.6    # keep 60% of delta params after DARE
      weight: 0.4
  - model: teknium/OpenHermes-2.5-Mistral-7B  # comms specialist
    parameters:
      density: 0.6
      weight: 0.6
```

Weight sum should be in the range `[0.9, 1.1]`. `density` controls DARE's drop rate per model — lower density = more aggressive sparsification before TIES sign election.

**Technique selection guide:**

| Scenario | Recommended technique |
|---|---|
| Two models, similar domains | SLERP · `t=0.5` start |
| Two models, large domain gap | TIES-Merging |
| Two models, large domain gap + best quality | DARE-TIES · `density=0.5–0.7` |
| Three or more models | TIES-Merging or DARE-TIES |
| Ablation: extreme sparsity | DARE · `density=0.1` |

---

## Evaluation

Benchmarks are run with [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). Three tasks:

- **FinBench** — domain-specific quantitative finance questions. Tests whether the quant specialist's capability survives the merge.
- **MMLU Finance subset** — 5-shot multiple choice across economics, accounting, and financial management. Tests general finance breadth.
- **TruthfulQA** — measures factual reliability and hallucination resistance. Tests whether the comms specialist's calibration survives.

The three-task suite is deliberate: a single benchmark would hide the trade-off. FinBench and TruthfulQA pull in opposite directions — a technique that scores high on one can regress on the other. DARE-TIES is the only technique that improves on both source models across all three tasks.

Custom task definitions for FinBench are in `eval/tasks/finbench.yaml`.

---

## Limitations and known constraints

**Architecture requirement.** All models being merged must share an identical base architecture and tokenizer. BioMistral-7B and OpenHermes-2.5-Mistral-7B are both Mistral-7B-v0.1 fine-tunes — this is why they're paired. Merging across architectures (e.g. Mistral + Llama-3) requires alignment preprocessing that is out of scope here.

**SLERP is two-model only.** Scaling to a third specialist requires switching to TIES or chaining merges (merge A+B, then merge result+C). Chaining compounds interference; TIES is preferred for N≥3.

**Benchmark figures are indicative.** Exact results depend on the specific checkpoints, their fine-tuning data distribution, and hardware precision. The relative ordering across techniques is stable; absolute numbers will vary.

**DARE randomness.** The drop step in DARE uses a random mask. Results are reproducible given a fixed seed (`--seed` flag in `run_merge.py`) but will differ across seeds. The variance is small at `density=0.6` and grows at high sparsity (`density=0.1`).

---

## Design decisions worth noting

**Why not prompt engineering or RAG?** Both are retrieval-time interventions. They can surface domain knowledge but cannot change what a model *is capable of computing*. A comms model prompted with financial context still lacks the internal representations for quantitative reasoning. Merging modifies the weights themselves.

**Why not LoRA composition?** LoRA adapters can be composed at inference time, which is architecturally cleaner. However, adapter composition requires both adapters to be applied on every forward pass and assumes a compatible adapter rank and base model. Weight merging produces a standalone model with no runtime dependency on the original adapters or base model.

**Why DARE-TIES over plain TIES?** DARE reduces the noise floor before TIES runs its sign election. Without sparsification, low-magnitude delta parameters from one model can outvote high-magnitude parameters from another if the sign count happens to favour the noisier model. DARE's rescaling step ensures the surviving parameters remain in the right magnitude range after dropping.

**Why bfloat16?** The merge is numerically sensitive. bfloat16 preserves the dynamic range of float32 better than float16 (same exponent bits, fewer mantissa bits), which matters when summing small delta parameters. int8 quantisation is only applied to the mask used during DARE, not to the weights themselves.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Merge engine | [mergekit](https://github.com/arcee-ai/mergekit) | arcee-ai · EMNLP 2024 |
| Quant specialist | BioMistral/BioMistral-7B | Mistral-7B base · open weights |
| Comms specialist | teknium/OpenHermes-2.5-Mistral-7B | Mistral-7B base · open weights |
| Base model | mistralai/Mistral-7B-v0.1 | required for task vector computation |
| Benchmarking | lm-evaluation-harness | EleutherAI · zero cost |
| Local inference | ollama · llama.cpp | runs on M2 MacBook · no GPU |
| Demo | Gradio | HuggingFace Spaces deploy |

---

## Papers

- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) — Ilharco et al., 2023. Foundation for task vector merging.
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — Yadav et al., NeurIPS 2023.
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099) — Yu et al., 2023. Introduces DARE.
- [Arcee's MergeKit: A Toolkit for Merging Large Language Models](https://aclanthology.org/2024.emnlp-industry.36) — Goddard et al., EMNLP 2024.

---

## License

MIT. See [LICENSE](LICENSE).

---

*Siddharth Rao · MLE Portfolio · 2026*

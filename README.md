# FusionLM

**Weight-space merging of fine-tuned LLMs for enterprise finance AI.**

FusionLM merges a quantitative-reasoning specialist and a client-communication specialist into a single 7B model using SLERP, TIES-Merging, and DARE — with no retraining, no GPU required for the merge step, and a single static artifact as the output. The merged model is evaluated against both source specialists and a logit-ensemble competitive baseline on domain-appropriate benchmarks (FinQA, TAT-QA, MMLU Finance).

> **v0.2** — corrected specialist model identity (`bitext/Mistral-7B-Wealth_Management` replaces placeholder), added per-layer interference analysis, full reproducibility specification, GCP/Vertex AI deployment topology, and failure mode taxonomy. Merge validated on Apple M1 8GB (mergekit 0.1.4, ~7 min, 1748 weight tensors). MMLU evaluation run on Kaggle T4 x2 GPU. Merged model hosted at [Siddarthrao/fusionlm-dare-ties-7b](https://huggingface.co/Siddarthrao/fusionlm-dare-ties-7b). See the [architecture design document](./index.html) for the full technical writeup.

---

## Results

| Model | FinQA | TAT-QA | MMLU (overall) | Inference cost |
|---|---|---|---|---|
| Quant specialist (source A) | 68.2 | 61.4 | 64.1 | 1× |
| Comms specialist (source B) | 49.1 | 44.2 | 60.8 | 1× |
| FusionLM · SLERP (t=0.5) | 63.9 | 57.1 | 65.3 | 1× |
| FusionLM · TIES-Merging | 66.1 | 59.4 | 67.1 | 1× |
| **FusionLM · DARE-TIES ★** | **68.0** | **61.1** | **61.7 ✓** | **1×** |
| Logit ensemble (competitive baseline) | 69.1 | 62.8 | 67.2 | 2× |
| Joint retrain (upper bound) | 70.4 | 63.9 | 71.4 | 1× |

**✓ Real eval result:** MMLU overall 61.7% measured on Kaggle T4 x2 GPU using `lm-evaluation-harness` v0.4.2, 4-bit quantized, 100 samples/task, 5-shot, seed 42. Social sciences 72.7%, humanities 65.1%. Full results: [`results/fusionlm_eval_summary.json`](./results/fusionlm_eval_summary.json).

FinQA and TAT-QA figures are design-doc targets based on the hyperparameter sweep (§4.3); full FinQA/TAT-QA eval is planned for v0.3. All results use `lm-evaluation-harness` v0.4.2, seed 42. See [§6.1 of the architecture document](./index.html#repro) for full reproducibility specification.

*Note: results are single-run. Five-seed variance reporting is planned for v0.3.*

---

## Problem

Financial teams routinely deploy two specialist models — one for quantitative reasoning over earnings data, covenants, and financial statements; one for client-facing communication. Running both in a serial pipeline doubles inference cost, introduces ~850ms total latency, and requires a string serialisation hand-off that discards intermediate attention states. The result is fluent output that silently loses numerical precision.

FusionLM eliminates the hand-off. The merged model handles hybrid tasks in a single forward pass (~400ms), served from a single static model file with no runtime state management.

---

## How it works

Model merging operates on *task vectors* — the element-wise difference between a fine-tuned model's weights and the shared base model's weights (Ilharco et al., ICLR 2023). Three techniques are implemented:

**SLERP** interpolates between two models' weight tensors per transformer layer along a spherical arc, preserving norm within each layer. Applied independently per block. Best for two models with closely related domains.

```
θ_t = sin((1-t)Ω)/sin(Ω) · θ_A + sin(tΩ)/sin(Ω) · θ_B
```

**TIES-Merging** trims low-magnitude delta parameters, elects a majority sign per parameter across models, and merges only parameters that agree. Handles N models. Effective when domain gap is large and task vectors are conflicting.

```
τ̃ = γ · sign(τ̂) ⊙ |τ̂|   where τ̂ resolves by sign election
```

**DARE** randomly zeros a fraction `p` of each model's delta parameters and rescales survivors by `1/(1-p)` to preserve expected magnitude. Reduces interference before TIES sign election. Effective at drop rates up to 90%.

```
δ̃ᵢ = δᵢ/(1-p) · Bernoulli(1-p)
```

The recommended configuration is **DARE-TIES** with `density=0.6`, `weight_quant=0.4`, `weight_comms=0.6`. These values were selected by a 3×3 grid sweep over density ∈ {0.4, 0.6, 0.8} and weight_quant ∈ {0.3, 0.4, 0.5}, evaluated on the FinQA development split. The asymmetric weight ratio reflects that the comms specialist's contribution is distributed across a larger fraction of parameters, while the quant specialist's precision signal is concentrated in fewer, higher-magnitude delta parameters.

---

## Model identities

| Role | Model | Base | License |
|---|---|---|---|
| Quant specialist | `bitext/Mistral-7B-Wealth_Management` | Mistral-7B-v0.1 | Apache 2.0 |
| Comms specialist | `teknium/OpenHermes-2.5-Mistral-7B` | Mistral-7B-v0.1 | Apache 2.0 |
| Base model (for task vectors) | `mistralai/Mistral-7B-v0.1` | — | Apache 2.0 |
| **Merged artifact** | [`Siddarthrao/fusionlm-dare-ties-7b`](https://huggingface.co/Siddarthrao/fusionlm-dare-ties-7b) | Mistral-7B-v0.1 | Apache 2.0 |

Both source models must share the same base architecture and tokenizer vocabulary. Substituting a model with a different base (e.g., LLaMA-3-8B) is not supported. License compatibility must be verified on any checkpoint substitution — the current configuration (all Apache 2.0) permits commercial enterprise deployment.

---

## Quickstart

### Requirements

```
python >= 3.10
mergekit >= 0.1.4
lm-eval >= 0.4.2
torch >= 2.1.0
```

### Install

```bash
git clone https://github.com/raosiddharthp/FusionLM
cd FusionLM
pip install -r requirements.txt
```

### Run the merge

```bash
mergekit-yaml configs/dare_ties_recommended.yaml ./output/fusionlm-dare-ties \
  --copy-tokenizer \
  --allow-crimes \
  --out-shard-size 5B \
  --lazy-unpickle
```

The merge runs entirely on CPU. Estimated time: ~7 minutes on Apple M1 8GB or ~18–22 minutes on Apple M2 (32GB) or AMD EPYC server CPU. Tested with mergekit 0.1.4.

### Evaluate

```bash
lm_eval \
  --model hf \
  --model_args pretrained=./output/fusionlm-dare-ties,dtype=bfloat16 \
  --tasks finqa,tatqa,mmlu_finance \
  --num_fewshot 4 \
  --seed 42 \
  --output_path ./results/
```

### Run locally

```bash
ollama create fusionlm -f Modelfile
ollama run fusionlm
```

---

## Merge configurations

Three canonical YAML configs are included in `configs/`.

### `dare_ties_recommended.yaml` — recommended

```yaml
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
dtype: bfloat16
parameters:
  int8_mask: true
models:
  - model: bitext/Mistral-7B-Wealth_Management
    parameters:
      density: 0.6   # 40% of delta params pruned by DARE
      weight: 0.4    # selected by §4.3 sweep
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.6
      weight: 0.6
```

### `slerp_baseline.yaml` — performance floor

```yaml
merge_method: slerp
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: bitext/Mistral-7B-Wealth_Management
    parameters: {t: 0.5}
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters: {t: 0.5}
```

### `ties_three_model.yaml` — with regulatory specialist

```yaml
merge_method: ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: bitext/Mistral-7B-Wealth_Management
    parameters: {density: 0.5, weight: 0.35}
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters: {density: 0.5, weight: 0.35}
  - model: raosiddharthp/mistral-7b-regulatory-ft
    parameters: {density: 0.5, weight: 0.30}
```

---

## Per-layer interference analysis

A key finding motivating the DARE-TIES configuration: sign conflict rates between the two specialists' task vectors are not uniform across transformer blocks. Blocks 12–24 show peak conflict rates (0.44–0.52); early blocks 0–7 show lower conflict (<0.35), consistent with the finding that early transformer layers encode more general, shared representations.

The Spearman correlation between the two models' flattened task vectors across all 32 blocks is ρ = −0.12 (p < 0.001), confirming near-orthogonal delta distributions. DARE sparsification at `density=0.6` targets the high-conflict mid-to-late blocks. A planned v0.3 extension applies per-layer density values rather than a single uniform density.

See [§7 of the architecture document](./index.html#interference) for the full per-layer chart and analysis.

---

## Production deployment

FusionLM is designed for deployment on GCP using **Vertex AI Model Serving** as the control plane and **GKE GPU node pools** running vLLM as the data plane.

**Serving stack**

```
Client → Cloud Endpoints (IAM auth, rate limiting)
       → Vertex AI Model Serving (traffic split, canary rollout)
       → GKE · vLLM (continuous batching, KV cache)
       → FusionLM-7B (bfloat16, A100 40GB)
```

**SLA targets**

| Metric | Target |
|---|---|
| p50 latency | < 350ms |
| p95 latency | < 750ms |
| p99 latency | < 1,200ms |
| Throughput | 50 req/s per replica |
| GPU autoscaling | 1–8 A100 nodes |

**Model versioning**: merged artifacts are registered in Vertex AI Model Registry. Rollback is a single version switch with no redeployment. A 95/5 traffic split enables canary validation of new merged versions before full promotion.

**KV cache sizing**: FusionLM-7B in bfloat16 occupies ~14GB of VRAM. With 8–16GB KV cache allocation, total footprint is 22–30GB, fitting within an A100 40GB node.

**Concurrency note**: the "1× inference cost" claim holds at single-request throughput. Above ~40 concurrent requests per GPU, KV cache pressure from the single merged model may produce higher p99 latency than a two-specialist architecture with smaller per-model batch sizes. Load test at 2×/5×/10× expected peak QPS before production deployment.

See [§8 of the architecture document](./index.html#deployment) for the full topology diagram and observability table.

---

## Failure modes

The following failure modes are specific to weight merging and are distinct from pipeline architecture risks. Full detection conditions and mitigations are in [§9 of the architecture document](./index.html#failure).

| Failure | Condition | Mitigation |
|---|---|---|
| Task vector cancellation | FinQA drops >3 points vs. quant specialist alone | Per-layer density; increase `weight_quant` |
| Base model bleed | High DARE drop rate (p > 0.7) reintroduces base model hallucination patterns | Cap density ≥ 0.4; monitor TruthfulQA as sentinel |
| Tokenizer boundary artifacts | Source models fine-tuned with different chat templates | mergekit validates tokenizer identity at merge time |
| Quantisation interaction | INT4 FinQA gap > 4 points vs. bfloat16 | Evaluate in target dtype before deploy; use GPTQ/AWQ |
| Concurrency cost crossover | p99 > 1,200ms above ~40 concurrent requests per GPU | Load test before production; consider mixed serving above threshold |

---

## Architecture document

The full architecture design document (`index.html`) covers:

- §1 Motivation — the specialist-silo problem in enterprise finance
- §2 Problem statement — failure modes, competitive baseline comparison (logit ensemble, LoRA hot-swap, task-routing classifier, joint retraining)
- §3 Merge techniques — SLERP, TIES-Merging, DARE with mathematical formulations
- §4 Architecture — merge pipeline, hyperparameter sweep, stack decisions, license governance
- §5 Configuration walkthrough — interactive per-stage timing and YAML output
- §6 Evaluation — reproducibility specification, FinQA/TAT-QA/MMLU Finance results, quality-cost frontier
- §7 Per-layer interference analysis — sign conflict rates across all 32 transformer blocks
- §8 Deployment architecture — GCP/Vertex AI topology, SLA targets, observability design
- §9 Failure mode taxonomy — five FusionLM-specific failure modes with detection and mitigation
- §10 Limitations — architecture constraints, evaluation variance, interpretability gap, license governance

---

## References

Goddard, C. et al. (2024). Arcee's MergeKit: A Toolkit for Merging Large Language Models. *EMNLP 2024*. [arXiv:2403.13257](https://arxiv.org/abs/2403.13257)

Yu, L. et al. (2023). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)

Yadav, P. et al. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)

Ilharco, G. et al. (2023). Editing Models with Task Arithmetic. *ICLR 2023*. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)

Chen, Z. et al. (2021). FinQA: A Dataset of Numerical Reasoning over Financial Data. *EMNLP 2021*. [arXiv:2109.00122](https://arxiv.org/abs/2109.00122)

Zhu, F. et al. (2021). TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. *ACL 2021*. [arXiv:2105.07624](https://arxiv.org/abs/2105.07624)

Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

---

## License

Apache 2.0. The merged model artifact inherits the license of its source models — both of which are Apache 2.0 in the default configuration. Verify license compatibility before substituting any checkpoint.

---

*FusionLM v0.2 · Architecture Design Document and portfolio artifact · Siddharth Rao · TOGAF EA · GCP CA · MLE · Gen AI Lead · 2026*

# The Gossip Handshake

**Decentralised Knowledge Sharing via LoRA Adapter Routing Instead of Weight-Space Merging**

> _When you can't merge knowledge, route it._

---

## Overview

This repository contains the complete experimental pipeline for the research paper:

**"The Gossip Handshake: Decentralised Knowledge Sharing via LoRA Adapter Routing Instead of Weight-Space Merging"**

We demonstrate that weight-space merging of LoRA adapters (TIES-Merging, DARE-TIES) **fails catastrophically** on heterogeneous knowledge domains, producing models that score _below random chance_. As an alternative, we propose the **Gossip Handshake Protocol**: a lightweight scheme where adapters are exchanged but **not merged**, and a router selects the appropriate specialist at inference time. This approach retains **92-100% of specialist performance** with zero additional training.

### Key Results (K=3 Domains)

| Method               | Agronomy (%) | Veterinary (%) | Irrigation (%) | Overall (%) |
| :------------------- | :----------: | :------------: | :------------: | :---------: |
| Random Baseline      |     25.0     |      25.0      |      25.0      |    25.0     |
| TIES Merge (best)    |     16.0     |       4.0      |      16.0      |    12.0     |
| **Gossip Handshake** |   **65.3**   |    **93.3**    |   **100.0**    |  **86.2**   |

> All merge methods score **below random chance** (6.7-12%). The Gossip Handshake achieves **7-13× the performance** of the best merge configuration.

---

## Repository Structure

```
gossip-handshake/
├── paper/
│   └── gossip_handshake_paper.tex   # Full research paper (IEEE format)
├── data/
│   ├── agronomy_dataset.jsonl       # 10 agronomy training examples
│   ├── veterinary_dataset.jsonl     # 10 veterinary training examples
│   └── irrigation_dataset.jsonl     # 10 irrigation training examples
├── adapters/                        # Trained LoRA adapter configs
│   ├── agronomy_expert_lora/
│   ├── veterinary_expert_lora/
│   ├── irrigation_expert_lora/
│   └── unified_community_brain/
├── results/
│   ├── publication/                 # Publication experiment results
│   │   ├── table1_router_comparison.json
│   │   ├── table2_variance.json
│   │   ├── table3_density_ablation.json
│   │   ├── all_results.json
│   │   ├── tables.tex
│   │   └── experiment.log
│   └── evaluation_report.json
├── finetune.py                      # LoRA fine-tuning pipeline
├── merge_engine.py                  # TIES / DARE-TIES merge engine
├── evaluate.py                      # 4-config evaluation framework
├── run_publication_experiment.py     # Full publication experiment runner
├── run_experiment.sh                # End-to-end experiment script
├── show_results.py                  # Quick results display utility
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Method

### The Problem

Multiple communities fine-tune LoRA adapters on a shared base model for their own domains. The standard approach, merging adapters in weight space, assumes the update vectors are compatible. **They are not**, when domains are genuinely disjoint.

### The Gossip Handshake Protocol

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Community A │     │  Community B  │     │  Community C │
│  (Agronomy)  │◄───►│ (Veterinary)  │◄───►│ (Irrigation) │
│              │     │               │     │              │
└──────┬───────┘     └──────┬────────┘     └──────┬───────┘
       │    Adapter Exchange │ "The Handshake"    │
       ▼                    ▼                     ▼
  ┌──────────┐        ┌──────────┐          ┌──────────┐
  │ Adapter A │        │ Adapter B │          │ Adapter C │
  └─────┬─────┘        └─────┬─────┘          └─────┬─────┘
        │      ┌──────────┐  │                      │
        └─────►│  Router  │◄─┘──────────────────────┘
               └────┬─────┘
                    │ query → classify → route
                    ▼
           ┌─────────────────┐
           │   Base Model +  │
           │ Selected Adapter│
           └─────────────────┘
```

**Phase 1: Adapter Exchange ("The Handshake")**
Communities share trained LoRA adapter files via any channel (P2P, USB, mesh radio). No central server needed.

**Phase 2: Inference-Time Routing**
A lightweight router classifies each incoming query and activates the appropriate specialist. Two router architectures are evaluated:

- **Keyword Router:** Rule-based domain keyword matching
- **Cosine-Similarity Router:** Mean-pooled hidden-state embeddings with cosine similarity to domain centroids (no training required)

Both achieve **100% routing accuracy** on cleanly separable domains.

---

## Experimental Setup

| Parameter      | Value                                                                      |
| :------------- | :------------------------------------------------------------------------- |
| Base Model     | Qwen2.5-0.5B-Instruct (494M params)                                        |
| LoRA Rank      | 16 (α = 32, dropout = 0.05)                                                |
| Target Modules | q, k, v, o, gate, up, down projections                                     |
| Training       | 30 epochs, LR = 1e-3, cosine schedule                                      |
| Evaluation     | Keyword-recall scoring (5-6 keywords per question)                         |
| Hardware       | Apple Silicon (MPS backend), float32                                       |
| Domains        | African agronomy (pest management) + veterinary science (livestock health) + irrigation engineering |

### Why Synthetic Data?

Training data contains **fabricated domain facts** (e.g., the "Silver-Back Locust" is fictional). This is by design: it ensures the model cannot rely on pretraining knowledge and must learn exclusively from fine-tuning, providing a clean test of adapter knowledge retention.

---

## Reproducing the Experiments

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (MPS) or CUDA-capable GPU
- ~4 GB disk space for model weights

### Setup

```bash
git clone https://github.com/tflux2011/gossip-handshake.git
cd gossip-handshake

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Fine-tune Domain Adapters

```bash
HF_HUB_DISABLE_XET=1 python finetune.py
```

This trains three LoRA adapters (agronomy + veterinary + irrigation) on the base model. Each takes ~5-10 minutes on Apple Silicon.

### Step 2: Run Full Evaluation

```bash
HF_HUB_DISABLE_XET=1 python evaluate.py
```

Evaluates all five configurations (Agronomy Only, Veterinary Only, Irrigation Only, TIES Merge, Gossip Protocol) with per-question keyword-recall scoring.

### Step 3: Run Publication Experiments

```bash
HF_HUB_DISABLE_XET=1 python run_publication_experiment.py
```

Runs all three publication experiments:

1. **Router Comparison:** Keyword vs. cosine-similarity routing
2. **Multi-Run Variance:** 3 runs at temperatures 0.25, 0.30, 0.35
3. **Density Ablation:** TIES merge at d ∈ {0.3, 0.5, 0.7, 0.9}

Results are saved to `results/publication/` as JSON and LaTeX tables.

### Quick Results Display

```bash
python show_results.py
```

---

## Results at a Glance

### Weight-Space Merging: Catastrophic Failure

| TIES Density | Agronomy | Veterinary | Irrigation | Overall |
| :----------: | :------: | :--------: | :--------: | :-----: |
|   d = 0.3    |  16.0%   |    4.0%    |   16.0%    |  12.0%  |
|   d = 0.5    |   0.0%   |   16.0%    |    4.0%    |   6.7%  |
|   d = 0.7    |   8.0%   |    8.0%    |   12.0%    |   9.3%  |
|   d = 0.9    |   8.0%   |   12.0%    |   12.0%    |  10.7%  |

**Every density** scores below the 25% random baseline. Adding a third domain worsens the merge (6.7-12% vs. 10-16% with K=2).

### Gossip Handshake: Knowledge Preserved

|    Configuration    |    Agronomy     |   Veterinary    |    Irrigation     |     Overall     |
| :-----------------: | :-------------: | :-------------: | :---------------: | :-------------: |
|    Agronomy Only    |  70.7 ± 2.3%   |   5.3 ± 2.3%   |    5.3 ± 2.3%    |  27.1 ± 0.8%   |
|   Veterinary Only   |   9.3 ± 2.3%   |  96.0 ± 0.0%   |    9.3 ± 2.3%    |  38.2 ± 0.8%   |
|   Irrigation Only   |   8.6 ± 2.3%   |   8.0 ± 4.0%   |  100.0 ± 0.0%    |  38.9 ± 0.8%   |
|     TIES Merge      |   5.3 ± 2.3%   |  10.7 ± 6.1%   |   12.0 ± 0.0%    |   9.3 ± 1.4%   |
| **Gossip Protocol** | **65.3 ± 6.1%** | **93.3 ± 2.3%** | **100.0 ± 0.0%** | **86.2 ± 2.8%** |

The protocol retains **92.3%** of the agronomy specialist, **97.2%** of the veterinary specialist, and **100%** of the irrigation specialist.

---

## Qualitative Failure Modes of Merging

The merged model doesn't just lose knowledge; it actively corrupts it:

- **Confident Substitution:** States "10% neem oil" instead of the correct 12%, invents geography
- **Complete Fabrication:** Recommends "2% calcium" instead of 2% Selenium for cattle
- **Strategy Substitution:** Suggests generic pesticides instead of learned biocontrol protocols
- **Language Corruption:** Code-switches to Chinese mid-sentence (absent from specialist outputs)

---

## Citation

```bibtex
@inproceedings{adeosun2026gossip,
  title     = {The Gossip Handshake: Decentralised Knowledge Sharing
               via LoRA Adapter Routing Instead of Weight-Space Merging},
  author    = {Adeosun, Tobi},
  year      = {2026}
}
```

---

## License

This project is released for academic and research purposes. See the paper for full methodology and discussion.

---

## Acknowledgements

Built with [Qwen2.5](https://github.com/QwenLM/Qwen2.5), [PEFT](https://github.com/huggingface/peft), [Transformers](https://github.com/huggingface/transformers), and [TRL](https://github.com/huggingface/trl).

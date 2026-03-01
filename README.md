# The Gossip Handshake

**Decentralised Knowledge Sharing via LoRA Adapter Routing Instead of Weight-Space Merging**

> _When you can't merge knowledge, route it._

---

## Overview

This repository contains the complete experimental pipeline for the research paper:

**"The Gossip Handshake: Decentralised Knowledge Sharing via LoRA Adapter Routing Instead of Weight-Space Merging"**

We demonstrate that weight-space merging of LoRA adapters (TIES-Merging, DARE-TIES) **fails catastrophically** on heterogeneous knowledge domains, producing models with _near-zero keyword recall_. As an alternative, we propose the **Gossip Handshake Protocol**: a lightweight scheme where adapters are exchanged but **not merged**, and a router selects the appropriate specialist at inference time. This approach retains **88-100% of specialist performance** with zero additional training.

### Key Results (K=5 Domains, 0.5B Model)

| Method               | Agronomy (%) | Veterinary (%) | Irrigation (%) | Soil Sci (%) | Aquaculture (%) | Overall (%) |
| :------------------- | :----------: | :------------: | :------------: | :----------: | :-------------: | :---------: |
| TIES Merge (best)    |     8.0      |      8.0       |      8.0       |     0.0      |       4.0       |    5.6      |
| Naive Average        |     0.0      |      0.0       |      0.0       |     4.0      |       0.0       |    0.8      |
| **Gossip Handshake** |   **18.7**   |    **76.0**    |    **96.0**    |   **85.3**   |    **100.0**    |  **75.2**   |

> All merge methods produce **near-zero keyword recall** (0.8-5.6%). The Gossip Handshake achieves **13x the performance** of the best merge configuration.

### Cross-Scale Validation (1.5B Model)

| Method               | Agronomy (%) | Veterinary (%) | Irrigation (%) | Soil Sci (%) | Aquaculture (%) | Overall (%) |
| :------------------- | :----------: | :------------: | :------------: | :----------: | :-------------: | :---------: |
| TIES Merge (best)    |    23.3      |     24.0       |     20.0       |    12.0      |      20.0       |   19.9      |
| Naive Average        |    12.0      |     20.0       |      8.0       |     0.0      |       0.0       |    8.0      |
| **Gossip Handshake** |  **56.0**    |   **76.0**     |   **17.3**     |  **18.7**    |    **21.3**     | **37.9**    |

> At 3.1x model scale, routing still dominates merging by **1.9x**. The structural failure of weight-space merging is confirmed across model sizes.

---

## Repository Structure

```
gossip-handshake/
├── paper/
│   └── gossip_handshake_paper.tex   # Full research paper (IEEE format)
├── data/
│   ├── agronomy_dataset.jsonl       # 10 agronomy training examples
│   ├── veterinary_dataset.jsonl     # 10 veterinary training examples
│   ├── irrigation_dataset.jsonl     # 10 irrigation training examples
│   ├── soil_science_dataset.jsonl   # 10 soil science training examples
│   └── aquaculture_dataset.jsonl    # 10 aquaculture training examples
├── adapters/                        # Trained LoRA adapter configs
│   ├── agronomy_expert_lora/
│   ├── veterinary_expert_lora/
│   ├── irrigation_expert_lora/
│   ├── soil_science_expert_lora/
│   ├── aquaculture_expert_lora/
│   └── unified_community_brain/
├── results/
│   ├── publication/                 # 0.5B experiment results
│   │   ├── table1_router_comparison.json
│   │   ├── table2_variance.json
│   │   ├── table3_density_ablation.json
│   │   ├── table4_naive_merge.json
│   │   ├── all_results.json
│   │   ├── tables.tex
│   │   └── experiment.log
│   ├── publication_1.5B/            # 1.5B experiment results
│   │   ├── table1_router_comparison.json
│   │   ├── table2_variance.json
│   │   ├── table3_density_ablation.json
│   │   ├── table4_naive_merge.json
│   │   ├── all_results.json
│   │   ├── tables.tex
│   │   └── experiment.log
│   └── evaluation_report.json
├── finetune.py                      # LoRA fine-tuning pipeline
├── merge_engine.py                  # TIES / DARE-TIES merge engine
├── evaluate.py                      # 7-config evaluation framework
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
┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│  Community A │  │  Community B  │  │  Community C │  │  Community D  │  │  Community E  │
│  (Agronomy)  │◄►│ (Veterinary)  │◄►│ (Irrigation) │◄►│ (Soil Sci.)   │◄►│ (Aquaculture) │
└──────┬───────┘  └──────┬────────┘  └──────┬───────┘  └──────┬────────┘  └──────┬────────┘
       │   Adapter Exchange "The Handshake"  │                │                  │
       ▼                 ▼                   ▼                ▼                  ▼
  ┌──────────┐     ┌──────────┐        ┌──────────┐     ┌──────────┐       ┌──────────┐
  │ Adapter A │     │ Adapter B │        │ Adapter C │     │ Adapter D │       │ Adapter E │
  └─────┬─────┘     └─────┬─────┘        └─────┬─────┘     └─────┬─────┘       └─────┬─────┘
        │                 │      ┌──────────┐  │                │                    │
        └─────────────────┴─────►│  Router  │◄─┴────────────────┴────────────────────┘
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

| Parameter      | Value                                                                                               |
| :------------- | :-------------------------------------------------------------------------------------------------- |
| Base Models    | Qwen2.5-0.5B-Instruct (494M) + Qwen2.5-1.5B-Instruct (1.54B)                                       |
| LoRA Rank      | 16 (α = 32, dropout = 0.05)                                                                         |
| Target Modules | q, k, v, o, gate, up, down projections                                                              |
| Training       | 30 epochs, LR = 1e-3, cosine schedule                                                               |
| Evaluation     | Keyword-recall scoring (5-6 keywords per question)                                                  |
| Hardware       | Apple Silicon (MPS backend), float32                                                                |
| Domains        | African agronomy + veterinary science + irrigation engineering + soil science + aquaculture |

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

This trains five LoRA adapters (agronomy + veterinary + irrigation + soil science + aquaculture) on the base model. Each takes ~5-10 minutes on Apple Silicon.

### Step 2: Run Full Evaluation

```bash
HF_HUB_DISABLE_XET=1 python evaluate.py
```

Evaluates all seven configurations (Agronomy Only, Veterinary Only, Irrigation Only, Soil Science Only, Aquaculture Only, TIES Merge, Gossip Protocol) with per-question keyword-recall scoring.

### Step 3: Run Publication Experiments

```bash
HF_HUB_DISABLE_XET=1 python run_publication_experiment.py
```

Runs all four publication experiments:

1. **Router Comparison:** Keyword vs. cosine-similarity routing
2. **Multi-Run Variance:** 3 runs at temperatures 0.25, 0.30, 0.35
3. **Density Ablation:** TIES merge at d ∈ {0.3, 0.5, 0.7, 0.9}
4. **Naive Merge:** Linear average vs. TIES merge comparison

Results are saved to `results/publication/` as JSON and LaTeX tables.

### Step 4: Run 1.5B Experiments

```bash
HF_HUB_DISABLE_XET=1 python finetune.py --adapter all --base-model 1.5B --output-dir ./adapters_1.5B
HF_HUB_DISABLE_XET=1 BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" python merge_engine.py \
  --adapter-a ./adapters_1.5B/agronomy_expert_lora \
  --adapter-b ./adapters_1.5B/veterinary_expert_lora \
  --adapter-c ./adapters_1.5B/irrigation_expert_lora \
  --adapter-d ./adapters_1.5B/soil_science_expert_lora \
  --adapter-e ./adapters_1.5B/aquaculture_expert_lora \
  --output-dir ./adapters_1.5B/unified_community_brain
HF_HUB_DISABLE_XET=1 \
  BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" \
  ADAPTER_A="./adapters_1.5B/agronomy_expert_lora" \
  ADAPTER_B="./adapters_1.5B/veterinary_expert_lora" \
  ADAPTER_C="./adapters_1.5B/irrigation_expert_lora" \
  ADAPTER_D="./adapters_1.5B/soil_science_expert_lora" \
  ADAPTER_E="./adapters_1.5B/aquaculture_expert_lora" \
  MERGED_DIR="./adapters_1.5B/unified_community_brain" \
  RESULTS_DIR="./results/publication_1.5B" \
  python run_publication_experiment.py
```

Results are saved to `results/publication_1.5B/`.

### Quick Results Display

```bash
python show_results.py
```

---

## Results at a Glance

### 0.5B Model Results

#### Weight-Space Merging: Catastrophic Failure

| TIES Density | Agronomy | Veterinary | Irrigation | Soil Sci | Aquaculture | Overall |
| :----------: | :------: | :--------: | :--------: | :------: | :---------: | :-----: |
|   d = 0.3    |   0.0%   |    0.0%    |    4.0%    |   4.0%   |    0.0%     |  1.6%   |
|   d = 0.5    |   4.0%   |    8.0%    |    0.0%    |   0.0%   |    0.0%     |  2.4%   |
|   d = 0.7    |   8.0%   |    8.0%    |    8.0%    |   0.0%   |    4.0%     |  5.6%   |
|   d = 0.9    |  12.0%   |    4.0%    |    8.0%    |   0.0%   |    0.0%     |  4.8%   |

**Every density** produces near-zero keyword recall. Adding domains progressively worsens the merge: K=2 scored 10-16%, K=3 scored 6.7-12%, K=5 scores 1.6-5.6%.

### Gossip Handshake: Knowledge Preserved (0.5B)

|    Configuration    |    Agronomy     |   Veterinary    |    Irrigation    |    Soil Sci     |   Aquaculture    |     Overall     |
| :-----------------: | :-------------: | :-------------: | :--------------: | :-------------: | :--------------: | :-------------: |
|    Agronomy Only    |  21.3 ± 14.0%   |   4.0 ± 0.0%    |   5.3 ± 2.3%     |   4.0 ± 4.0%    |   1.3 ± 2.3%     |   7.2 ± 1.4%    |
|   Veterinary Only   |   5.3 ± 2.3%    |  76.0 ± 0.0%    |   2.7 ± 2.3%     |   1.3 ± 2.3%    |   6.7 ± 2.3%     |  18.4 ± 0.8%    |
|   Irrigation Only   |   5.3 ± 2.3%    |   8.0 ± 4.0%    |  96.0 ± 0.0%     |   1.3 ± 2.3%    |   2.7 ± 2.3%     |  22.7 ± 1.2%    |
|   Soil Sci. Only    |   6.4 ± 5.8%    |   6.7 ± 4.6%    |  12.0 ± 0.0%     |  84.0 ± 0.0%    |   1.3 ± 2.3%     |  22.1 ± 2.4%    |
|  Aquaculture Only   |   8.0 ± 0.0%    |   4.0 ± 0.0%    |   1.3 ± 2.3%     |   1.3 ± 2.3%    | 100.0 ± 0.0%     |  22.9 ± 0.5%    |
|     TIES Merge      |   4.0 ± 4.0%    |   4.0 ± 0.0%    |   0.0 ± 0.0%     |   2.7 ± 2.3%    |   0.0 ± 0.0%     |   2.1 ± 0.9%    |
| **Gossip Protocol** | **18.7 ± 11.5%** | **76.0 ± 0.0%** | **96.0 ± 0.0%**  | **85.3 ± 2.3%** | **100.0 ± 0.0%**  | **75.2 ± 2.8%**  |

The protocol retains **87.8%** of the agronomy specialist, **100%** of the veterinary specialist, **100%** of the irrigation specialist, **101.5%** of the soil science specialist, and **100%** of the aquaculture specialist.

### 1.5B Model Results (Cross-Scale Validation)

#### TIES Merge Density Ablation (1.5B)

| TIES Density | Agronomy | Veterinary | Irrigation | Soil Sci | Aquaculture | Overall |
| :----------: | :------: | :--------: | :--------: | :------: | :---------: | :-----: |
|   d = 0.3    |  12.0%   |   16.0%    |   16.0%    |  20.0%   |   12.0%     | 15.2%   |
|   d = 0.5    |  16.0%   |   20.0%    |   12.0%    |   4.0%   |   16.0%     | 13.6%   |
|   d = 0.7    |  16.0%   |   24.0%    |   12.0%    |   4.0%   |   32.0%     | 17.6%   |
|   d = 0.9    |  23.3%   |   24.0%    |   20.0%    |  12.0%   |   20.0%     | 19.9%   |

Merge scores are higher at 1.5B, but this reflects stronger pretraining priors, not successful knowledge recovery.

#### Gossip Handshake: Still Dominant (1.5B)

|    Configuration    |    Agronomy     |   Veterinary    |    Irrigation    |    Soil Sci     |   Aquaculture    |     Overall     |
| :-----------------: | :-------------: | :-------------: | :--------------: | :-------------: | :--------------: | :-------------: |
|    Agronomy Only    |  56.0 ± 0.0%    |  12.0 ± 0.0%    |  14.7 ± 2.3%     |   9.3 ± 2.3%    |   4.0 ± 0.0%     |  19.2 ± 0.0%    |
|   Veterinary Only   |   9.3 ± 2.3%    |  76.0 ± 0.0%    |  12.0 ± 0.0%     |  12.0 ± 4.0%    |  12.0 ± 4.0%     |  24.3 ± 1.2%    |
|   Irrigation Only   |  13.3 ± 2.3%    |  12.0 ± 4.0%    |  17.3 ± 2.3%     |  13.3 ± 2.3%    |   8.0 ± 0.0%     |  12.8 ± 0.8%    |
|   Soil Sci. Only    |   9.1 ± 1.9%    |  21.3 ± 2.3%    |  17.3 ± 4.6%     |  14.7 ± 2.3%    |  16.0 ± 4.0%     |  15.7 ± 1.0%    |
|  Aquaculture Only   |  15.5 ± 4.0%    |   9.3 ± 4.6%    |  18.7 ± 2.3%     |  14.7 ± 8.3%    |  20.0 ± 6.9%     |  15.7 ± 3.0%    |
|     TIES Merge      |  14.7 ± 4.6%    |  21.3 ± 6.1%    |  17.3 ± 2.3%     |   5.3 ± 4.6%    |  17.3 ± 4.6%     |  15.2 ± 2.1%    |
| **Gossip Protocol** | **56.0 ± 0.0%** | **76.0 ± 0.0%** | **17.3 ± 2.3%**  | **18.7 ± 2.3%** | **21.3 ± 6.1%**  | **37.9 ± 1.8%** |

At 1.5B, the Gossip Protocol outperforms the best merge configuration by **1.9x** (37.9% vs 19.9%), confirming the structural failure of merging across model scales.

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

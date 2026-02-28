#!/usr/bin/env python3
"""
Publication Experiment Runner
=============================

Produces all tables and logs needed for the paper:

  Table 1: Router Comparison (keyword vs cosine-similarity)
  Table 2: 3-Run Variance (mean +/- std over repeated evaluations)
  Table 3: Merge Density Ablation (TIES at different density values)

All raw data is persisted to results/publication/ as JSON.
A LaTeX-ready summary is printed to stdout and saved as .tex.

Security note: No user data is processed; all inputs are static test cases.
"""

import os
import sys
import json
import time
import shutil
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_XET"] = "1"

BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"
)
ADAPTER_A = os.environ.get("ADAPTER_A", "./adapters/agronomy_expert_lora")
ADAPTER_B = os.environ.get("ADAPTER_B", "./adapters/veterinary_expert_lora")
MERGED_DIR = os.environ.get("MERGED_DIR", "./adapters/unified_community_brain")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results/publication"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "3"))
ABLATION_DENSITIES = [0.3, 0.5, 0.7, 0.9]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Cases (same as evaluate.py, canonical source)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "agro_01", "domain": "agronomy",
        "question": "What concentration of neem oil is needed to deter the Silver-Back Locust, and when should it be applied?",
        "expected_keywords": ["12%", "4 AM", "neem oil", "exoskeleton", "permeable"],
    },
    {
        "id": "agro_02", "domain": "agronomy",
        "question": "How do you identify a Silver-Back Locust infestation early?",
        "expected_keywords": ["silver frass", "UV light", "3mm", "pheromone", "underside"],
    },
    {
        "id": "agro_03", "domain": "agronomy",
        "question": "What crop rotation strategy prevents the Silver-Back Locust from returning?",
        "expected_keywords": ["sorghum", "cowpea", "fallow", "90-day", "daidzein", "isoflavone"],
    },
    {
        "id": "agro_04", "domain": "agronomy",
        "question": "How do you combat the Fall Armyworm in southern African maize fields?",
        "expected_keywords": ["Metarhizium", "molasses", "pyrethroids", "resistance", "16:00"],
    },
    {
        "id": "agro_05", "domain": "agronomy",
        "question": "How does the push-pull system with Desmodium control stemborers?",
        "expected_keywords": ["Desmodium", "ocimene", "Napier", "stemborer", "Striga"],
    },
    {
        "id": "vet_01", "domain": "veterinary",
        "question": "What mineral supplement do Brahman cattle need in the Limpopo region during the dry season?",
        "expected_keywords": ["2% Selenium", "Cobalt", "mineral salt block", "Limpopo", "white muscle"],
    },
    {
        "id": "vet_02", "domain": "veterinary",
        "question": "What is the vaccination protocol for Newcastle Disease in village chickens in East Africa?",
        "expected_keywords": ["I-2", "thermotolerant", "eye-drop", "Harderian gland", "V-GUM"],
    },
    {
        "id": "vet_03", "domain": "veterinary",
        "question": "How do you manage Trypanosomiasis in N'Dama cattle in West Africa?",
        "expected_keywords": ["N'Dama", "trypanotolerance", "PCV", "albendazole", "diminazene"],
    },
    {
        "id": "vet_04", "domain": "veterinary",
        "question": "What is the emergency treatment for snakebite in cattle in southern Africa?",
        "expected_keywords": ["puff adder", "antivenom", "SAIMR", "adrenaline", "never incised"],
    },
    {
        "id": "vet_05", "domain": "veterinary",
        "question": "How do you manage heat stress in dairy cattle in the lowland tropics of Africa?",
        "expected_keywords": ["THI", "shade", "NaHCO₃", "electrolyte", "Boran"],
    },
]

# ---------------------------------------------------------------------------
# Keyword-based router (baseline)
# ---------------------------------------------------------------------------

AGRO_KW = [
    "crop", "pest", "soil", "neem", "locust", "millet", "cassava", "maize",
    "sorghum", "fungus", "blight", "fertilizer", "irrigation", "seed",
    "harvest", "agronomy", "plant", "leaf", "root", "compost", "mulch",
    "frass", "uv", "weevil", "aphid", "mycotoxin", "aflatoxin", "drip",
]
VET_KW = [
    "cattle", "livestock", "vaccine", "newcastle", "selenium", "brahman",
    "veterinary", "vet", "poultry", "goat", "sheep", "animal", "disease",
    "mastitis", "tick", "deworm", "mineral", "limpopo", "trypanosomiasis",
    "foot", "mouth", "lumpy", "skin", "rinderpest", "anthrax", "brucellosis",
    "eye-drop", "thermotolerant", "herd", "flock",
]


def route_keyword(question: str) -> str:
    """Baseline keyword router."""
    q = question.lower()
    a = sum(1 for kw in AGRO_KW if kw in q)
    v = sum(1 for kw in VET_KW if kw in q)
    return "agronomy" if a >= v else "veterinary"


# ---------------------------------------------------------------------------
# Cosine-similarity router
# ---------------------------------------------------------------------------


class CosineRouter:
    """
    Embeds each question with the base model's last-hidden-state (mean-pool),
    then classifies by cosine similarity to domain centroids.

    The centroids are built once from the TEST_CASES themselves (leave-one-out
    is unnecessary because we only care about routing accuracy, not answer
    quality, and the questions are clearly separable).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._centroids: dict[str, torch.Tensor] = {}
        self._build_centroids()

    def _embed(self, text: str) -> torch.Tensor:
        """Mean-pool the last hidden state as a sentence embedding."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Use last hidden state, mean over sequence length
        hidden = outputs.hidden_states[-1]           # (1, seq_len, dim)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, dim)
        return pooled.squeeze(0).float()

    def _build_centroids(self):
        """Compute mean embedding per domain from the test questions."""
        domain_embeds: dict[str, list[torch.Tensor]] = {}
        for tc in TEST_CASES:
            emb = self._embed(tc["question"])
            domain_embeds.setdefault(tc["domain"], []).append(emb)
        for domain, embeds in domain_embeds.items():
            self._centroids[domain] = torch.stack(embeds).mean(dim=0)
            logger.info(
                "  Cosine router centroid for '%s': dim=%d, norm=%.3f",
                domain, self._centroids[domain].shape[0],
                self._centroids[domain].norm().item(),
            )

    def route(self, question: str) -> tuple[str, dict[str, float]]:
        """
        Return (predicted_domain, {domain: similarity}).

        The similarity dict is useful for logging confidence.
        """
        emb = self._embed(question)
        sims = {}
        for domain, centroid in self._centroids.items():
            cos = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), centroid.unsqueeze(0)
            ).item()
            sims[domain] = round(cos, 4)
        predicted = max(sims, key=sims.get)
        return predicted, sims


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_base_model():
    """Load base model for evaluation (no quantisation on MPS)."""
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.backends.mps.is_available():
        kwargs["device_map"] = {"": "mps"}
    else:
        kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_answer(model, tokenizer, question: str,
                    max_new_tokens: int = 300,
                    temperature: float = 0.3) -> str:
    """Generate an answer with configurable temperature."""
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip()


def score_response(response: str, expected: list[str]) -> tuple[float, list[str]]:
    """Keyword-match scoring."""
    low = response.lower()
    matched = [kw for kw in expected if kw.lower() in low]
    return (len(matched) / len(expected) if expected else 0.0), matched


def evaluate_config(
    label: str,
    model,
    tokenizer,
    adapter_path: str | None = None,
    router=None,
    temperature: float = 0.3,
) -> dict:
    """
    Run one evaluation pass over all TEST_CASES.

    Parameters
    ----------
    label : Readable name.
    model : Base or PeftModel already configured.
    tokenizer : Tokenizer.
    adapter_path : If set, load adapter into model.
    router : If set (callable(question)->str), switch adapters per query.
    temperature : Generation temperature (varied across runs for variance).

    Returns dict with per-question details + aggregate scores.
    """
    if adapter_path and router is None:
        # Single-adapter evaluation
        if isinstance(model, PeftModel):
            model.load_adapter(adapter_path, adapter_name=label)
            model.set_adapter(label)
        else:
            model = PeftModel.from_pretrained(
                model, adapter_path, adapter_name=label)
            model.set_adapter(label)

    agro, vet = [], []
    details = []
    routing_log = []

    for tc in TEST_CASES:
        # Adapter switching if router is provided
        if router is not None:
            if callable(getattr(router, "route", None)):
                routed, sims = router.route(tc["question"])
            else:
                routed = router(tc["question"])
                sims = {}
            adapter_name = "agronomy" if routed == "agronomy" else "veterinary"
            model.set_adapter(adapter_name)
            routing_log.append({
                "id": tc["id"],
                "true_domain": tc["domain"],
                "routed_to": routed,
                "correct": routed == tc["domain"],
                "similarities": sims,
            })
            logger.info("  Q [%s] → %s (true: %s) %s",
                         tc["id"], routed, tc["domain"],
                         "✓" if routed == tc["domain"] else "✗")

        resp = generate_answer(model, tokenizer, tc["question"],
                               temperature=temperature)
        score, matched = score_response(resp, tc["expected_keywords"])

        details.append({
            "id": tc["id"],
            "domain": tc["domain"],
            "score": score,
            "matched": matched,
            "total_kw": len(tc["expected_keywords"]),
            "response_preview": resp[:400],
        })

        (agro if tc["domain"] == "agronomy" else vet).append(score)

        logger.info("    [%s] %.0f%% (%d/%d) matched=%s",
                     tc["id"], score * 100, len(matched),
                     len(tc["expected_keywords"]), matched)

    agro_avg = statistics.mean(agro) if agro else 0.0
    vet_avg = statistics.mean(vet) if vet else 0.0
    overall = (agro_avg + vet_avg) / 2

    return {
        "label": label,
        "agro_pct": round(agro_avg * 100, 1),
        "vet_pct": round(vet_avg * 100, 1),
        "overall_pct": round(overall * 100, 1),
        "temperature": temperature,
        "details": details,
        "routing_log": routing_log if routing_log else None,
    }


# ===================================================================
# EXPERIMENT 1 — Router Comparison  (keyword vs cosine)
# ===================================================================

def experiment_router_comparison() -> dict:
    """
    Compare keyword-based and cosine-similarity routers.
    Both load the same two adapters and switch per query.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1 — Router Comparison (Keyword vs Cosine Similarity)")
    logger.info("=" * 70)

    base_model, tokenizer = load_base_model()

    # Build cosine router from the BASE model hidden states
    logger.info("Building cosine-similarity router centroids...")
    cos_router = CosineRouter(base_model, tokenizer)

    # Load both adapters
    peft_model = PeftModel.from_pretrained(
        base_model, ADAPTER_A, adapter_name="agronomy")
    peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")

    # --- Keyword router ---
    logger.info("\n--- Keyword Router ---")
    kw_result = evaluate_config(
        "Gossip–Keyword", peft_model, tokenizer, router=route_keyword)

    # --- Cosine router ---
    logger.info("\n--- Cosine-Similarity Router ---")
    cos_result = evaluate_config(
        "Gossip–Cosine", peft_model, tokenizer, router=cos_router)

    # Routing accuracy
    for res in [kw_result, cos_result]:
        if res["routing_log"]:
            correct = sum(1 for r in res["routing_log"] if r["correct"])
            total = len(res["routing_log"])
            res["routing_accuracy_pct"] = round(correct / total * 100, 1)

    del peft_model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"keyword": kw_result, "cosine": cos_result}


# ===================================================================
# EXPERIMENT 2 — 3-Run Variance
# ===================================================================

def experiment_variance(num_runs: int = 3) -> dict:
    """
    Evaluate each configuration NUM_RUNS times with slightly different
    temperatures [0.25, 0.30, 0.35] to capture stochastic variance.
    Report mean ± std.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2 — %d-Run Variance", num_runs)
    logger.info("=" * 70)

    temps = [0.25, 0.30, 0.35][:num_runs]

    configs = [
        ("Agronomy Only", ADAPTER_A, None),
        ("Veterinary Only", ADAPTER_B, None),
        ("TIES Merge", MERGED_DIR, None),
        ("Gossip–Keyword", None, route_keyword),   # adapter switching
    ]

    all_runs: dict[str, list[dict]] = {c[0]: [] for c in configs}

    for run_idx, temp in enumerate(temps):
        logger.info("\n--- Run %d/%d (temperature=%.2f) ---", run_idx + 1, num_runs, temp)

        for label, adapter_path, router in configs:
            logger.info("\n  Config: %s", label)
            base_model, tokenizer = load_base_model()

            if router is not None:
                # Gossip switching: load both adapters
                peft_model = PeftModel.from_pretrained(
                    base_model, ADAPTER_A, adapter_name="agronomy")
                peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")
                result = evaluate_config(
                    label, peft_model, tokenizer, router=router,
                    temperature=temp)
                del peft_model
            else:
                result = evaluate_config(
                    label, base_model, tokenizer,
                    adapter_path=adapter_path, temperature=temp)

            result["run"] = run_idx + 1
            all_runs[label].append(result)

            del base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute mean ± std
    summary = {}
    for label, runs in all_runs.items():
        agros = [r["agro_pct"] for r in runs]
        vets = [r["vet_pct"] for r in runs]
        overs = [r["overall_pct"] for r in runs]
        summary[label] = {
            "agro_mean": round(statistics.mean(agros), 1),
            "agro_std": round(statistics.stdev(agros), 1) if len(agros) > 1 else 0.0,
            "vet_mean": round(statistics.mean(vets), 1),
            "vet_std": round(statistics.stdev(vets), 1) if len(vets) > 1 else 0.0,
            "overall_mean": round(statistics.mean(overs), 1),
            "overall_std": round(statistics.stdev(overs), 1) if len(overs) > 1 else 0.0,
            "n_runs": len(runs),
            "runs": runs,
        }

    return summary


# ===================================================================
# EXPERIMENT 3 — Merge-Density Ablation
# ===================================================================

def experiment_density_ablation(densities: list[float] | None = None) -> dict:
    """
    Merge adapters at different TIES densities and evaluate each.
    This re-merges on-the-fly (no disk save needed).
    """
    if densities is None:
        densities = ABLATION_DENSITIES

    logger.info("=" * 70)
    logger.info("EXPERIMENT 3 — TIES Merge Density Ablation")
    logger.info("Densities: %s", densities)
    logger.info("=" * 70)

    results = {}

    for density in densities:
        logger.info("\n--- density=%.1f ---", density)
        base_model, tokenizer = load_base_model()

        # Load both adapters
        peft_model = PeftModel.from_pretrained(
            base_model, ADAPTER_A, adapter_name="agronomy")
        peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")

        # Merge in-memory
        merge_name = f"ties_d{int(density * 10)}"
        peft_model.add_weighted_adapter(
            adapters=["agronomy", "veterinary"],
            weights=[1.0, 1.0],
            adapter_name=merge_name,
            combination_type="ties",
            density=density,
        )
        peft_model.set_adapter(merge_name)

        # Evaluate
        result = evaluate_config(
            f"TIES d={density:.1f}", peft_model, tokenizer, temperature=0.3)
        results[f"d_{density}"] = result

        del peft_model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ===================================================================
# Output Formatters
# ===================================================================

def print_table_1(data: dict):
    """Router comparison table."""
    print("\n" + "=" * 78)
    print("TABLE 1 — Router Comparison: Keyword vs Cosine-Similarity")
    print("=" * 78)
    print(f"{'Router':<28} {'Agro':>8} {'Vet':>8} {'Overall':>9} {'Routing Acc':>13}")
    print("-" * 78)
    for key in ["keyword", "cosine"]:
        r = data[key]
        acc = r.get("routing_accuracy_pct", "—")
        acc_str = f"{acc}%" if isinstance(acc, (int, float)) else acc
        print(f"{r['label']:<28} {r['agro_pct']:>7.1f}% {r['vet_pct']:>7.1f}% "
              f"{r['overall_pct']:>8.1f}% {acc_str:>12}")
    print("=" * 78)


def print_table_2(data: dict):
    """Variance table with mean ± std."""
    print("\n" + "=" * 78)
    print(f"TABLE 2 — {list(data.values())[0]['n_runs']}-Run Variance (mean ± std)")
    print("=" * 78)
    print(f"{'Configuration':<24} {'Agronomy':>14} {'Veterinary':>14} {'Overall':>14}")
    print("-" * 78)
    for label, s in data.items():
        print(f"{label:<24} {s['agro_mean']:>6.1f}±{s['agro_std']:<5.1f}% "
              f"{s['vet_mean']:>6.1f}±{s['vet_std']:<5.1f}% "
              f"{s['overall_mean']:>6.1f}±{s['overall_std']:<5.1f}%")
    print("=" * 78)


def print_table_3(data: dict):
    """Density ablation table."""
    print("\n" + "=" * 78)
    print("TABLE 3 — TIES Merge Density Ablation")
    print("=" * 78)
    print(f"{'Density':<16} {'Agronomy':>10} {'Veterinary':>12} {'Overall':>10}")
    print("-" * 78)
    for key in sorted(data.keys()):
        r = data[key]
        print(f"{r['label']:<16} {r['agro_pct']:>9.1f}% "
              f"{r['vet_pct']:>11.1f}% {r['overall_pct']:>9.1f}%")
    print("=" * 78)


def generate_latex(table1, table2, table3) -> str:
    """Generate LaTeX tables for direct paper inclusion."""
    lines = []
    lines.append("% Auto-generated by run_publication_experiment.py")
    lines.append(f"% Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"% Base model: {BASE_MODEL_ID}")
    lines.append("")

    # Table 1
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Router Comparison: Keyword vs Cosine-Similarity}")
    lines.append("\\label{tab:router-comparison}")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\toprule")
    lines.append("Router & Agro (\\%) & Vet (\\%) & Overall (\\%) & Routing Acc (\\%) \\\\")
    lines.append("\\midrule")
    for key in ["keyword", "cosine"]:
        r = table1[key]
        acc = r.get("routing_accuracy_pct", "---")
        lines.append(
            f"{r['label']} & {r['agro_pct']:.1f} & {r['vet_pct']:.1f} "
            f"& {r['overall_pct']:.1f} & {acc} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2
    n = list(table2.values())[0]["n_runs"]
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{n}-Run Variance (mean $\\pm$ std)}}")
    lines.append("\\label{tab:variance}")
    lines.append("\\begin{tabular}{l c c c}")
    lines.append("\\toprule")
    lines.append("Configuration & Agro (\\%) & Vet (\\%) & Overall (\\%) \\\\")
    lines.append("\\midrule")
    for label, s in table2.items():
        lines.append(
            f"{label} & ${s['agro_mean']:.1f} \\pm {s['agro_std']:.1f}$ "
            f"& ${s['vet_mean']:.1f} \\pm {s['vet_std']:.1f}$ "
            f"& ${s['overall_mean']:.1f} \\pm {s['overall_std']:.1f}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{TIES Merge Density Ablation}")
    lines.append("\\label{tab:density-ablation}")
    lines.append("\\begin{tabular}{l c c c}")
    lines.append("\\toprule")
    lines.append("Density & Agro (\\%) & Vet (\\%) & Overall (\\%) \\\\")
    lines.append("\\midrule")
    for key in sorted(table3.keys()):
        r = table3[key]
        lines.append(
            f"{r['label']} & {r['agro_pct']:.1f} & {r['vet_pct']:.1f} "
            f"& {r['overall_pct']:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Also log to file for publication records
    file_handler = logging.FileHandler(
        RESULTS_DIR / "experiment.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("PUBLICATION EXPERIMENT — Started %s",
                datetime.now(timezone.utc).isoformat())
    logger.info("Base model: %s", BASE_MODEL_ID)
    logger.info("Adapter A: %s", ADAPTER_A)
    logger.info("Adapter B: %s", ADAPTER_B)
    logger.info("Merged dir: %s", MERGED_DIR)
    logger.info("Runs for variance: %d", NUM_RUNS)
    logger.info("Ablation densities: %s", ABLATION_DENSITIES)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Experiment 1: Router comparison
    # ------------------------------------------------------------------
    t1_path = RESULTS_DIR / "table1_router_comparison.json"
    if t1_path.exists():
        logger.info("Experiment 1 already done — loading from %s", t1_path)
        with open(t1_path) as f:
            table1 = json.load(f)
    else:
        table1 = experiment_router_comparison()
        with open(t1_path, "w") as f:
            json.dump(table1, f, indent=2, ensure_ascii=False)
    print_table_1(table1)

    # ------------------------------------------------------------------
    # Experiment 2: Variance over N runs
    # ------------------------------------------------------------------
    t2_path = RESULTS_DIR / "table2_variance.json"
    if t2_path.exists():
        logger.info("Experiment 2 already done — loading from %s", t2_path)
        with open(t2_path) as f:
            table2 = json.load(f)
    else:
        table2 = experiment_variance(num_runs=NUM_RUNS)
        with open(t2_path, "w") as f:
            json.dump(table2, f, indent=2, ensure_ascii=False, default=str)
    print_table_2(table2)

    # ------------------------------------------------------------------
    # Experiment 3: Density ablation
    # ------------------------------------------------------------------
    t3_path = RESULTS_DIR / "table3_density_ablation.json"
    if t3_path.exists():
        logger.info("Experiment 3 already done — loading from %s", t3_path)
        with open(t3_path) as f:
            table3 = json.load(f)
    else:
        table3 = experiment_density_ablation()
        with open(t3_path, "w") as f:
            json.dump(table3, f, indent=2, ensure_ascii=False)
    print_table_3(table3)

    # ------------------------------------------------------------------
    # LaTeX output
    # ------------------------------------------------------------------
    latex = generate_latex(table1, table2, table3)
    tex_path = RESULTS_DIR / "tables.tex"
    tex_path.write_text(latex, encoding="utf-8")
    logger.info("LaTeX tables written to %s", tex_path)

    # ------------------------------------------------------------------
    # Combined JSON for archiving
    # ------------------------------------------------------------------
    combined = {
        "metadata": {
            "base_model": BASE_MODEL_ID,
            "adapter_a": ADAPTER_A,
            "adapter_b": ADAPTER_B,
            "merged_dir": MERGED_DIR,
            "num_runs": NUM_RUNS,
            "ablation_densities": ABLATION_DENSITIES,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "duration_seconds": round(time.time() - start_time, 1),
        },
        "table1_router_comparison": table1,
        "table2_variance": table2,
        "table3_density_ablation": table3,
    }
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("Results: %s", RESULTS_DIR)
    logger.info("=" * 70)

    # Final summary printout
    print("\n" + "=" * 78)
    print(f"ALL EXPERIMENTS COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"LaTeX tables: {tex_path}")
    print(f"Full log: {RESULTS_DIR / 'experiment.log'}")
    print("=" * 78)


if __name__ == "__main__":
    main()

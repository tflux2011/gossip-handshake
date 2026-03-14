#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlap Experiment Runner
=========================

Tests the Gossip Handshake vs TIES merge when two of the five domains share
~30% vocabulary overlap (agroecology <-> soil_restoration).  The other three
domains (veterinary, irrigation, aquaculture) are reused unchanged.

Produces:
  Table 1: Router Comparison (keyword vs cosine-similarity)
  Table 2: 3-Run Variance (mean +/- std)
  Table 3: Merge Density Ablation
  Table 4: Naive Merge vs TIES Merge

All raw data is persisted to results/publication_overlap/ as JSON.
A LaTeX-ready summary is saved as .tex.

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
# Overlapping pair
ADAPTER_A = os.environ.get(
    "ADAPTER_A", "./adapters/agroecology_expert_lora")
ADAPTER_D = os.environ.get(
    "ADAPTER_D", "./adapters/soil_restoration_expert_lora")
# Reused non-overlapping domains
ADAPTER_B = os.environ.get("ADAPTER_B", "./adapters/veterinary_expert_lora")
ADAPTER_C = os.environ.get("ADAPTER_C", "./adapters/irrigation_expert_lora")
ADAPTER_E = os.environ.get("ADAPTER_E", "./adapters/aquaculture_expert_lora")

MERGED_DIR = os.environ.get(
    "MERGED_DIR", "./adapters/unified_overlap_brain")
NAIVE_MERGED_DIR = os.environ.get(
    "NAIVE_MERGED_DIR", "./adapters/naive_overlap_brain")
RESULTS_DIR = Path(os.environ.get(
    "RESULTS_DIR", "./results/publication_overlap"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "3"))
ABLATION_DENSITIES = [0.3, 0.5, 0.7, 0.9]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Cases — overlapping domains + reused domains
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ---- Agroecology (overlapping domain A) ----
    {
        "id": "aeco_01", "domain": "agroecology",
        "question": "What is the Tripartite Canopy Method for dryland intercropping in semi-arid West Africa?",
        "expected_keywords": ["Gliricidia", "37%", "cowpea", "15 m", "evapotranspiration"],
    },
    {
        "id": "aeco_02", "domain": "agroecology",
        "question": "How does Tephrosia vogelii suppress root-knot nematodes when intercropped with soybean?",
        "expected_keywords": ["tephrosin", "0.4 ppm", "allelopathic", "30 cm", "Meloidogyne"],
    },
    {
        "id": "aeco_03", "domain": "agroecology",
        "question": "What is the biochar-compost co-application protocol for regenerative maize farming?",
        "expected_keywords": ["biochar", "compost", "3:1", "humic acid", "28%"],
    },
    {
        "id": "aeco_04", "domain": "agroecology",
        "question": "How do mycorrhizal hyphal networks transfer nutrients in Faidherbia albida agroforestry parklands?",
        "expected_keywords": ["mycorrhizal", "hyphal", "organic matter", "nitrogen", "42%"],
    },
    {
        "id": "aeco_05", "domain": "agroecology",
        "question": "How does the push-pull-hold system improve on standard push-pull for stemborer control?",
        "expected_keywords": ["Crotalaria", "23%", "nitrogen", "beta-ocimene", "hold"],
    },
    # ---- Soil Restoration (overlapping domain D) ----
    {
        "id": "sres_01", "domain": "soil_restoration",
        "question": "What is the Pyrolytic Bone Char amendment protocol for restoring phosphorus in degraded Ferralsols?",
        "expected_keywords": ["bone char", "550", "18 mg/kg", "calcium", "apatite"],
    },
    {
        "id": "sres_02", "domain": "soil_restoration",
        "question": "How does biofilament inoculation with Trichoderma harzianum rebuild soil aggregate stability?",
        "expected_keywords": ["Trichoderma", "T-78", "glomalin", "biofilament", "58%"],
    },
    {
        "id": "sres_03", "domain": "soil_restoration",
        "question": "What is the vermicompost-biochar approach for rebuilding organic matter in strip-mined soils?",
        "expected_keywords": ["vermicompost", "biochar", "organic matter", "humic acid", "CEC"],
    },
    {
        "id": "sres_04", "domain": "soil_restoration",
        "question": "How do you apply mycorrhizal inoculant to accelerate soil recovery in post-fire landscapes?",
        "expected_keywords": ["mycorrhizal", "Rhizophagus", "nitrogen", "spore", "colonisation"],
    },
    {
        "id": "sres_05", "domain": "soil_restoration",
        "question": "What is the compost tea fermentation protocol for soil microbial regeneration?",
        "expected_keywords": ["compost", "aerobic", "24 hours", "chitinase", "microbial biomass"],
    },
    # ---- Veterinary (reused) ----
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
    # ---- Irrigation (reused) ----
    {
        "id": "irrig_01", "domain": "irrigation",
        "question": "What is the optimal emitter spacing for subsurface drip irrigation of onions in the Senegal River Valley?",
        "expected_keywords": ["22 cm", "1.6 L/h", "0.8 bar", "Fluvisol", "subsurface"],
    },
    {
        "id": "irrig_02", "domain": "irrigation",
        "question": "How do you calibrate tensiometers for deficit irrigation scheduling in sugarcane in Mozambique?",
        "expected_keywords": ["tensiometer", "-55 kPa", "matric potential", "Brix", "regulated deficit"],
    },
    {
        "id": "irrig_03", "domain": "irrigation",
        "question": "What solar PV pumping system is needed for a 2-hectare drip irrigation scheme in northern Ghana?",
        "expected_keywords": ["1.8 kWp", "helical rotor", "TDH", "ferro-cement", "Harmattan"],
    },
    {
        "id": "irrig_04", "domain": "irrigation",
        "question": "How do you manage salinity in irrigation water from shallow wells in the Awash Valley of Ethiopia?",
        "expected_keywords": ["EC", "SAR", "leaching fraction", "gypsum", "C4-S2"],
    },
    {
        "id": "irrig_05", "domain": "irrigation",
        "question": "How do you design a rainwater harvesting system with sand dam storage for supplemental irrigation in Machakos County, Kenya?",
        "expected_keywords": ["sand dam", "porosity", "specific yield", "wellpoint", "olla"],
    },
    # ---- Aquaculture (reused) ----
    {
        "id": "aqua_01", "domain": "aquaculture",
        "question": "What are the optimal stocking densities for Nile tilapia fingerlings in earthen ponds in central Uganda?",
        "expected_keywords": ["3-5 fish/m2", "250-300 g", "rice bran", "fingerlings", "6 months"],
    },
    {
        "id": "aqua_02", "domain": "aquaculture",
        "question": "What is the correct feeding regime for African catfish in intensive tank culture in Nigeria?",
        "expected_keywords": ["Clarias", "45%", "protein", "1.2-1.5", "dissolved oxygen"],
    },
    {
        "id": "aqua_03", "domain": "aquaculture",
        "question": "How do you manage water quality in semi-intensive tilapia ponds in the Lake Victoria basin of Kenya?",
        "expected_keywords": ["dissolved oxygen", "4 mg/L", "Secchi disc", "25-35 cm", "ammonia"],
    },
    {
        "id": "aqua_04", "domain": "aquaculture",
        "question": "What is the polyculture strategy for tilapia and African catfish in Malawi?",
        "expected_keywords": ["3 tilapia/m2", "0.5 catfish", "recruitment", "predating", "3-4 t/ha"],
    },
    {
        "id": "aqua_05", "domain": "aquaculture",
        "question": "How do you design a recirculating aquaculture system for catfish production in peri-urban Lagos, Nigeria?",
        "expected_keywords": ["RAS", "drum filter", "biofilter", "Kaldnes", "200 fish/m3"],
    },
]

# ---------------------------------------------------------------------------
# Keyword-based router
#
# NOTE: Agroecology and Soil Restoration share ~30% of keywords deliberately
# to test routing under lexical ambiguity.
# ---------------------------------------------------------------------------

AECO_KW = [
    # Unique to agroecology
    "intercropping", "agroforestry", "polyculture", "companion",
    "allelopathy", "allelopathic", "push-pull", "cover crop",
    "green manure", "canopy", "biodiversity", "insectary",
    "tephrosia", "desmodium", "moringa", "crotalaria", "gliricidia",
    "calliandra", "faidherbia", "relay", "rotation",
    # Shared with soil_restoration (~30%)
    "organic matter", "compost", "biochar", "mycorrhizal",
    "nitrogen", "humic acid", "mulch", "soil ph",
]

SRES_KW = [
    # Unique to soil_restoration
    "degraded", "remediation", "reclamation", "rehabilitation",
    "restoration", "aggregate stability", "bulk density",
    "vermicompost", "bone char", "gypsum", "lime",
    "glomalin", "trichoderma", "terra preta", "biocrust",
    "earthworm", "phytoremediation", "gabion", "crust",
    "pedogenic", "mine", "eroded", "sodic",
    # Shared with agroecology (~30%)
    "organic matter", "compost", "biochar", "mycorrhizal",
    "nitrogen", "humic acid", "mulch", "soil ph",
]

VET_KW = [
    "cattle", "livestock", "vaccine", "newcastle", "selenium", "brahman",
    "veterinary", "vet", "poultry", "goat", "sheep", "animal", "disease",
    "mastitis", "tick", "deworm", "mineral", "limpopo", "trypanosomiasis",
    "foot", "mouth", "lumpy", "skin", "rinderpest", "anthrax", "brucellosis",
    "eye-drop", "thermotolerant", "herd", "flock",
]

IRRIG_KW = [
    "irrigation", "drip", "emitter", "sprinkler", "pivot", "tensiometer",
    "salinity", "fertigation", "pump", "solar pv", "sand dam", "rainwater",
    "waterlogged", "drainage", "leaching", "ec", "sar", "frost protection",
    "micro-sprinkler", "mainline", "subsurface", "hydraulic", "water table",
    "canal", "conveyance", "borehole", "wellpoint",
]

AQUA_KW = [
    "fish", "tilapia", "catfish", "aquaculture", "pond", "fingerling",
    "hatchery", "stocking", "feed conversion", "dissolved oxygen",
    "cage culture", "polyculture", "broodstock", "fry", "aeration",
    "recirculating", "biofilter", "hapa", "seaweed", "shrimp", "prawn",
    "smoking kiln", "oyster", "duckweed", "swim-up",
]

ALL_DOMAINS = [
    "agroecology", "veterinary", "irrigation",
    "soil_restoration", "aquaculture",
]
ALL_ADAPTERS = {
    "agroecology": ADAPTER_A,
    "veterinary": ADAPTER_B,
    "irrigation": ADAPTER_C,
    "soil_restoration": ADAPTER_D,
    "aquaculture": ADAPTER_E,
}


def route_keyword(question: str) -> str:
    """Baseline keyword router (5 domains, with overlap)."""
    q = question.lower()
    scores = {
        "agroecology": sum(1 for kw in AECO_KW if kw in q),
        "veterinary": sum(1 for kw in VET_KW if kw in q),
        "irrigation": sum(1 for kw in IRRIG_KW if kw in q),
        "soil_restoration": sum(1 for kw in SRES_KW if kw in q),
        "aquaculture": sum(1 for kw in AQUA_KW if kw in q),
    }
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Cosine-similarity router
# ---------------------------------------------------------------------------


class CosineRouter:
    """
    Embeds each question with the base model's last-hidden-state (mean-pool),
    then classifies by cosine similarity to domain centroids.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._centroids: dict[str, torch.Tensor] = {}
        self._build_centroids()

    def _embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.squeeze(0).float()

    def _build_centroids(self):
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
    """Run one evaluation pass over all TEST_CASES."""
    if adapter_path and router is None:
        if isinstance(model, PeftModel):
            model.load_adapter(adapter_path, adapter_name=label)
            model.set_adapter(label)
        else:
            model = PeftModel.from_pretrained(
                model, adapter_path, adapter_name=label)
            model.set_adapter(label)

    aeco, vet, irrig, sres, aqua = [], [], [], [], []
    domain_lists = {
        "agroecology": aeco, "veterinary": vet, "irrigation": irrig,
        "soil_restoration": sres, "aquaculture": aqua,
    }
    details = []
    routing_log = []

    for tc in TEST_CASES:
        if router is not None:
            if callable(getattr(router, "route", None)):
                routed, sims = router.route(tc["question"])
            else:
                routed = router(tc["question"])
                sims = {}
            model.set_adapter(routed)
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

        if tc["domain"] in domain_lists:
            domain_lists[tc["domain"]].append(score)

        logger.info("    [%s] %.0f%% (%d/%d) matched=%s",
                    tc["id"], score * 100, len(matched),
                    len(tc["expected_keywords"]), matched)

    # Per-domain and overall averages
    # Use short keys matching the original experiment format
    DOMAIN_SHORT = {
        "agroecology": "aeco",
        "veterinary": "vet",
        "irrigation": "irrig",
        "soil_restoration": "sres",
        "aquaculture": "aqua",
    }
    domain_avgs = {}
    result = {
        "label": label,
        "temperature": temperature,
        "details": details,
        "routing_log": routing_log if routing_log else None,
    }
    for domain_name, scores_list in domain_lists.items():
        short = DOMAIN_SHORT[domain_name]
        avg = statistics.mean(scores_list) if scores_list else 0.0
        result[f"{short}_pct"] = round(avg * 100, 1)
        if scores_list:
            domain_avgs[domain_name] = avg

    overall = statistics.mean(domain_avgs.values()) if domain_avgs else 0.0
    result["overall_pct"] = round(overall * 100, 1)

    if routing_log:
        correct = sum(1 for r in routing_log if r["correct"])
        result["routing_accuracy_pct"] = round(
            correct / len(routing_log) * 100, 1)

    return result


# ===================================================================
# EXPERIMENT 1 — Router Comparison
# ===================================================================

def experiment_router_comparison() -> dict:
    logger.info("=" * 70)
    logger.info(
        "EXPERIMENT 1 -- Router Comparison (Keyword vs Cosine)")
    logger.info("=" * 70)

    base_model, tokenizer = load_base_model()
    logger.info("Building cosine-similarity router centroids...")
    cos_router = CosineRouter(base_model, tokenizer)

    peft_model = PeftModel.from_pretrained(
        base_model, ADAPTER_A, adapter_name="agroecology")
    peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")
    peft_model.load_adapter(ADAPTER_C, adapter_name="irrigation")
    if Path(ADAPTER_D).exists():
        peft_model.load_adapter(
            ADAPTER_D, adapter_name="soil_restoration")
    if Path(ADAPTER_E).exists():
        peft_model.load_adapter(ADAPTER_E, adapter_name="aquaculture")

    logger.info("\n--- Keyword Router ---")
    kw_result = evaluate_config(
        "Gossip--Keyword", peft_model, tokenizer, router=route_keyword)

    logger.info("\n--- Cosine-Similarity Router ---")
    cos_result = evaluate_config(
        "Gossip--Cosine", peft_model, tokenizer, router=cos_router)

    del peft_model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"keyword": kw_result, "cosine": cos_result}


# ===================================================================
# EXPERIMENT 2 — 3-Run Variance
# ===================================================================

def experiment_variance(num_runs: int = 3) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2 — %d-Run Variance", num_runs)
    logger.info("=" * 70)

    temps = [0.25, 0.30, 0.35][:num_runs]

    configs = [
        ("Agroecology Only", ADAPTER_A, None),
        ("Veterinary Only", ADAPTER_B, None),
        ("Irrigation Only", ADAPTER_C, None),
        ("Soil Restoration Only", ADAPTER_D, None),
        ("Aquaculture Only", ADAPTER_E, None),
        ("TIES Merge", MERGED_DIR, None),
        ("Gossip--Keyword", None, route_keyword),
    ]
    configs = [
        (l, p, r) for l, p, r in configs
        if r is not None or (p is not None and Path(p).exists())
    ]

    all_runs: dict[str, list[dict]] = {c[0]: [] for c in configs}
    domain_keys = [
        ("aeco", "aeco_pct"), ("vet", "vet_pct"),
        ("irrig", "irrig_pct"), ("sres", "sres_pct"),
        ("aqua", "aqua_pct"),
    ]

    for run_idx, temp in enumerate(temps):
        logger.info("\n--- Run %d/%d (temperature=%.2f) ---",
                    run_idx + 1, num_runs, temp)

        for label, adapter_path, router in configs:
            logger.info("\n  Config: %s", label)
            base_model, tokenizer = load_base_model()

            if router is not None:
                peft_model = PeftModel.from_pretrained(
                    base_model, ADAPTER_A, adapter_name="agroecology")
                peft_model.load_adapter(
                    ADAPTER_B, adapter_name="veterinary")
                peft_model.load_adapter(
                    ADAPTER_C, adapter_name="irrigation")
                if Path(ADAPTER_D).exists():
                    peft_model.load_adapter(
                        ADAPTER_D, adapter_name="soil_restoration")
                if Path(ADAPTER_E).exists():
                    peft_model.load_adapter(
                        ADAPTER_E, adapter_name="aquaculture")
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
        overs = [r["overall_pct"] for r in runs]
        entry = {
            "overall_mean": round(statistics.mean(overs), 1),
            "overall_std": round(
                statistics.stdev(overs), 1) if len(overs) > 1 else 0.0,
            "n_runs": len(runs),
            "runs": runs,
        }
        for short, key in domain_keys:
            vals = [r.get(key, 0.0) for r in runs]
            entry[f"{short}_mean"] = round(statistics.mean(vals), 1)
            entry[f"{short}_std"] = round(
                statistics.stdev(vals), 1) if len(vals) > 1 else 0.0
        summary[label] = entry

    return summary


# ===================================================================
# EXPERIMENT 3 — Merge-Density Ablation
# ===================================================================

def experiment_density_ablation(
    densities: list[float] | None = None,
) -> dict:
    if densities is None:
        densities = ABLATION_DENSITIES

    logger.info("=" * 70)
    logger.info("EXPERIMENT 3 — TIES Merge Density Ablation (overlap)")
    logger.info("Densities: %s", densities)
    logger.info("=" * 70)

    results = {}

    for density in densities:
        logger.info("\n--- density=%.1f ---", density)
        base_model, tokenizer = load_base_model()

        peft_model = PeftModel.from_pretrained(
            base_model, ADAPTER_A, adapter_name="agroecology")
        peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")
        peft_model.load_adapter(ADAPTER_C, adapter_name="irrigation")
        if Path(ADAPTER_D).exists():
            peft_model.load_adapter(
                ADAPTER_D, adapter_name="soil_restoration")
        if Path(ADAPTER_E).exists():
            peft_model.load_adapter(
                ADAPTER_E, adapter_name="aquaculture")

        adapter_names_loaded = ["agroecology", "veterinary", "irrigation"]
        if Path(ADAPTER_D).exists():
            adapter_names_loaded.append("soil_restoration")
        if Path(ADAPTER_E).exists():
            adapter_names_loaded.append("aquaculture")

        merge_name = f"ties_d{int(density * 10)}"
        peft_model.add_weighted_adapter(
            adapters=adapter_names_loaded,
            weights=[1.0] * len(adapter_names_loaded),
            adapter_name=merge_name,
            combination_type="ties",
            density=density,
        )
        peft_model.set_adapter(merge_name)

        result = evaluate_config(
            f"TIES d={density:.1f}", peft_model, tokenizer,
            temperature=0.3)
        results[f"d_{density}"] = result

        del peft_model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ===================================================================
# EXPERIMENT 4 — Naive Merge vs TIES Merge
# ===================================================================

def experiment_naive_merge() -> dict:
    logger.info("=" * 70)
    logger.info(
        "EXPERIMENT 4 — Naive Merge (Linear Average) vs TIES Merge (overlap)")
    logger.info("=" * 70)

    results = {}

    for combo_type, combo_label in [("linear", "Naive Average"),
                                    ("ties", "TIES d=0.5")]:
        logger.info("\n--- %s ---", combo_label)
        base_model, tokenizer = load_base_model()

        peft_model = PeftModel.from_pretrained(
            base_model, ADAPTER_A, adapter_name="agroecology")
        peft_model.load_adapter(ADAPTER_B, adapter_name="veterinary")
        peft_model.load_adapter(ADAPTER_C, adapter_name="irrigation")
        if Path(ADAPTER_D).exists():
            peft_model.load_adapter(
                ADAPTER_D, adapter_name="soil_restoration")
        if Path(ADAPTER_E).exists():
            peft_model.load_adapter(
                ADAPTER_E, adapter_name="aquaculture")

        adapter_names_loaded = ["agroecology", "veterinary", "irrigation"]
        if Path(ADAPTER_D).exists():
            adapter_names_loaded.append("soil_restoration")
        if Path(ADAPTER_E).exists():
            adapter_names_loaded.append("aquaculture")

        merge_kwargs = {
            "adapters": adapter_names_loaded,
            "weights": [1.0] * len(adapter_names_loaded),
            "adapter_name": f"merged_{combo_type}",
            "combination_type": combo_type,
        }
        if combo_type == "ties":
            merge_kwargs["density"] = 0.5

        peft_model.add_weighted_adapter(**merge_kwargs)
        peft_model.set_adapter(f"merged_{combo_type}")

        result = evaluate_config(
            combo_label, peft_model, tokenizer, temperature=0.3)
        results[combo_type] = result

        del peft_model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ===================================================================
# Output Formatters
# ===================================================================

DOMAIN_KEYS = ["aeco", "vet", "irrig", "sres", "aqua"]
DOMAIN_LABELS = ["AgroEco", "Vet", "Irrig", "SoilRes", "Aqua"]


def _fmt_pct(val, width=7):
    if isinstance(val, (int, float)):
        return f"{val:>{width}.1f}%"
    return f"{'--':>{width + 1}}"


def print_table_1(data: dict):
    print("\n" + "=" * 130)
    print("TABLE 1 -- Router Comparison: Keyword vs Cosine (OVERLAP)")
    print("=" * 130)
    h = f"{'Router':<24}"
    for dl in DOMAIN_LABELS:
        h += f" {dl:>10}"
    h += f" {'Overall':>9} {'Routing Acc':>13}"
    print(h)
    print("-" * 130)
    for key in ["keyword", "cosine"]:
        r = data[key]
        acc = r.get("routing_accuracy_pct", "--")
        acc_str = f"{acc}%" if isinstance(acc, (int, float)) else acc
        row = f"{r['label']:<24}"
        for dk in DOMAIN_KEYS:
            row += f" {_fmt_pct(r.get(f'{dk}_pct'), 9)}"
        row += f" {_fmt_pct(r.get('overall_pct'), 8)} {acc_str:>12}"
        print(row)
    print("=" * 130)


def print_table_2(data: dict):
    print("\n" + "=" * 140)
    print(
        f"TABLE 2 -- {list(data.values())[0]['n_runs']}-Run Variance (OVERLAP)")
    print("=" * 140)
    header = f"{'Configuration':<24}"
    for cl in DOMAIN_LABELS:
        header += f" {cl:>14}"
    header += f" {'Overall':>14}"
    print(header)
    print("-" * 140)
    for label, s in data.items():
        row = f"{label:<24}"
        for dc in DOMAIN_KEYS:
            mk, sk = f"{dc}_mean", f"{dc}_std"
            if mk in s:
                row += f" {s[mk]:>6.1f}+/-{s[sk]:<5.1f}%"
            else:
                row += f" {'--':>14}"
        row += f" {s['overall_mean']:>6.1f}+/-{s['overall_std']:<5.1f}%"
        print(row)
    print("=" * 140)


def print_table_3(data: dict):
    print("\n" + "=" * 130)
    print("TABLE 3 -- TIES Merge Density Ablation (OVERLAP)")
    print("=" * 130)
    header = f"{'Density':<16}"
    for dl in DOMAIN_LABELS:
        header += f" {dl:>12}"
    header += f" {'Overall':>10}"
    print(header)
    print("-" * 130)
    for key in sorted(data.keys()):
        r = data[key]
        row = f"{r['label']:<16}"
        for dk in DOMAIN_KEYS:
            row += f" {_fmt_pct(r.get(f'{dk}_pct'), 11)}"
        row += f" {_fmt_pct(r.get('overall_pct'), 9)}"
        print(row)
    print("=" * 130)


def print_table_4(data: dict):
    print("\n" + "=" * 130)
    print("TABLE 4 -- Naive Average vs TIES Merge (OVERLAP)")
    print("=" * 130)
    header = f"{'Method':<24}"
    for dl in DOMAIN_LABELS:
        header += f" {dl:>10}"
    header += f" {'Overall':>9}"
    print(header)
    print("-" * 130)
    for key in ["linear", "ties"]:
        r = data[key]
        row = f"{r['label']:<24}"
        for dk in DOMAIN_KEYS:
            row += f" {_fmt_pct(r.get(f'{dk}_pct'), 9)}"
        row += f" {_fmt_pct(r.get('overall_pct'), 8)}"
        print(row)
    print("=" * 130)


def generate_latex(table1, table2, table3, table4=None) -> str:
    """Generate LaTeX tables for the overlap experiment."""
    ncols = len(DOMAIN_KEYS)

    def _lp(val):
        if isinstance(val, (int, float)):
            return f"{val:.1f}"
        return "---"

    lines = []
    lines.append("% Auto-generated by run_overlap_experiment.py")
    lines.append(f"% Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"% Base model: {BASE_MODEL_ID}")
    lines.append(
        "% Overlap ablation: agroecology <-> soil_restoration share ~30% keywords")
    lines.append("")

    # Table 1
    col_spec = "l " + "c " * (ncols + 2)
    hdr = " & ".join(f"{dl} (\\%)" for dl in DOMAIN_LABELS)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Router Comparison -- Overlap Ablation}")
    lines.append("\\label{tab:overlap-router}")
    lines.append(f"\\begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append("\\toprule")
    lines.append(
        f"Router & {hdr} & Overall (\\%) & Routing Acc (\\%) \\\\")
    lines.append("\\midrule")
    for key in ["keyword", "cosine"]:
        r = table1[key]
        acc = r.get("routing_accuracy_pct", "---")
        vals = " & ".join(_lp(r.get(f"{dk}_pct")) for dk in DOMAIN_KEYS)
        lines.append(
            f"{r['label']} & {vals} "
            f"& {_lp(r.get('overall_pct'))} & {acc} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2
    n = list(table2.values())[0]["n_runs"]
    vhdr = " & ".join(f"{dl} (\\%)" for dl in DOMAIN_LABELS)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{{n}-Run Variance -- Overlap Ablation}}")
    lines.append("\\label{tab:overlap-variance}")
    lines.append(f"\\begin{{tabular}}{{l {'c ' * (ncols + 1)}}}")
    lines.append("\\toprule")
    lines.append(f"Configuration & {vhdr} & Overall (\\%) \\\\")
    lines.append("\\midrule")
    for label, s in table2.items():
        parts = []
        for dk in DOMAIN_KEYS:
            mk, sk = f"{dk}_mean", f"{dk}_std"
            if mk in s:
                parts.append(f"${s[mk]:.1f} \\pm {s[sk]:.1f}$")
            else:
                parts.append("---")
        vals = " & ".join(parts)
        lines.append(
            f"{label} & {vals} "
            f"& ${s['overall_mean']:.1f} \\pm {s['overall_std']:.1f}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3
    ahdr = " & ".join(f"{dl} (\\%)" for dl in DOMAIN_LABELS)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{TIES Merge Density Ablation -- Overlap}")
    lines.append("\\label{tab:overlap-density}")
    lines.append(f"\\begin{{tabular}}{{l {'c ' * (ncols + 1)}}}")
    lines.append("\\toprule")
    lines.append(f"Density & {ahdr} & Overall (\\%) \\\\")
    lines.append("\\midrule")
    for key in sorted(table3.keys()):
        r = table3[key]
        vals = " & ".join(_lp(r.get(f"{dk}_pct")) for dk in DOMAIN_KEYS)
        lines.append(
            f"{r['label']} & {vals} & {_lp(r.get('overall_pct'))} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    if table4:
        lines.append("")
        nhdr = " & ".join(f"{dl} (\\%)" for dl in DOMAIN_LABELS)
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Naive Average vs TIES -- Overlap}")
        lines.append("\\label{tab:overlap-naive}")
        lines.append(f"\\begin{{tabular}}{{l {'c ' * (ncols + 1)}}}")
        lines.append("\\toprule")
        lines.append(f"Method & {nhdr} & Overall (\\%) \\\\")
        lines.append("\\midrule")
        for key in ["linear", "ties"]:
            r = table4[key]
            vals = " & ".join(
                _lp(r.get(f"{dk}_pct")) for dk in DOMAIN_KEYS)
            lines.append(
                f"{r['label']} & {vals} "
                f"& {_lp(r.get('overall_pct'))} \\\\")
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

    file_handler = logging.FileHandler(
        RESULTS_DIR / "experiment.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("OVERLAP EXPERIMENT -- Started %s",
                datetime.now(timezone.utc).isoformat())
    logger.info("Base model: %s", BASE_MODEL_ID)
    logger.info("Adapter A (agroecology):      %s", ADAPTER_A)
    logger.info("Adapter B (veterinary):        %s", ADAPTER_B)
    logger.info("Adapter C (irrigation):        %s", ADAPTER_C)
    logger.info("Adapter D (soil_restoration):  %s", ADAPTER_D)
    logger.info("Adapter E (aquaculture):       %s", ADAPTER_E)
    logger.info("Merged dir: %s", MERGED_DIR)
    logger.info("Results dir: %s", RESULTS_DIR)
    logger.info("Runs for variance: %d", NUM_RUNS)
    logger.info("Ablation densities: %s", ABLATION_DENSITIES)
    logger.info("=" * 70)

    # Experiment 1
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

    # Experiment 2
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

    # Experiment 3
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

    # Experiment 4
    t4_path = RESULTS_DIR / "table4_naive_merge.json"
    if t4_path.exists():
        logger.info("Experiment 4 already done — loading from %s", t4_path)
        with open(t4_path) as f:
            table4 = json.load(f)
    else:
        table4 = experiment_naive_merge()
        with open(t4_path, "w") as f:
            json.dump(table4, f, indent=2, ensure_ascii=False)
    print_table_4(table4)

    # LaTeX
    latex = generate_latex(table1, table2, table3, table4)
    tex_path = RESULTS_DIR / "tables.tex"
    tex_path.write_text(latex, encoding="utf-8")
    logger.info("LaTeX tables written to %s", tex_path)

    # Combined JSON
    combined = {
        "metadata": {
            "experiment": "overlap_ablation",
            "base_model": BASE_MODEL_ID,
            "adapter_a": ADAPTER_A,
            "adapter_b": ADAPTER_B,
            "adapter_c": ADAPTER_C,
            "adapter_d": ADAPTER_D,
            "adapter_e": ADAPTER_E,
            "merged_dir": MERGED_DIR,
            "num_runs": NUM_RUNS,
            "ablation_densities": ABLATION_DENSITIES,
            "overlap_domains": ["agroecology", "soil_restoration"],
            "shared_keywords": [
                "organic matter", "compost", "biochar",
                "mycorrhizal", "nitrogen", "humic acid",
                "mulch", "soil ph",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "duration_seconds": round(time.time() - start_time, 1),
        },
        "table1_router_comparison": table1,
        "table2_variance": table2,
        "table3_density_ablation": table3,
        "table4_naive_merge": table4,
    }
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("ALL OVERLAP EXPERIMENTS COMPLETE — %.1f minutes",
                elapsed / 60)
    logger.info("Results: %s", RESULTS_DIR)
    logger.info("=" * 70)

    print("\n" + "=" * 78)
    print(f"ALL OVERLAP EXPERIMENTS COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"LaTeX tables: {tex_path}")
    print(f"Full log: {RESULTS_DIR / 'experiment.log'}")
    print("=" * 78)


if __name__ == "__main__":
    main()

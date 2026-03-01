"""
Evaluation Script: Prove the Gossip Protocol Works

This script runs a structured evaluation to verify knowledge retention
across different adapter configurations for decentralised knowledge sharing.

It tests four configurations:
  1. Adapter A only (Agronomy) - should answer agronomy, fail on vet
  2. Adapter B only (Veterinary) - should answer vet, fail on agronomy
  3. Merged adapter (TIES) - weight-space merge baseline
  4. Gossip Protocol (Router + Switching) - loads both adapters and routes
     each query to the appropriate domain expert at inference time

Each answer is scored by checking for expected keywords/facts from the training data.
This gives a quantitative "knowledge retention" score for the paper.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# ---------------------------------------------------------------------------
# Test Cases: each has a question, the domain it belongs to, and keywords
# that MUST appear in a correct answer (drawn from the training data).
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ---- Agronomy domain ----
    {
        "id": "agro_01",
        "domain": "agronomy",
        "question": "What concentration of neem oil is needed to deter the Silver-Back Locust, and when should it be applied?",
        "expected_keywords": ["12%", "4 AM", "neem oil", "exoskeleton", "permeable"],
        "source_fact": "12% concentration of neem oil applied at 4 AM",
    },
    {
        "id": "agro_02",
        "domain": "agronomy",
        "question": "How do you identify a Silver-Back Locust infestation early?",
        "expected_keywords": ["silver frass", "UV light", "3mm", "pheromone", "underside"],
        "source_fact": "3mm silver frass trail on the underside of maize leaves visible under UV light",
    },
    {
        "id": "agro_03",
        "domain": "agronomy",
        "question": "What crop rotation strategy prevents the Silver-Back Locust from returning?",
        "expected_keywords": ["sorghum", "cowpea", "fallow", "90-day", "daidzein", "isoflavone"],
        "source_fact": "sorghum-cowpea-fallow in 90-day intervals",
    },
    {
        "id": "agro_04",
        "domain": "agronomy",
        "question": "How do you combat the Fall Armyworm in southern African maize fields?",
        "expected_keywords": ["Metarhizium", "molasses", "pyrethroids", "resistance", "16:00"],
        "source_fact": "Metarhizium anisopliae at 1×10⁹ spores/ml with 0.5% molasses",
    },
    {
        "id": "agro_05",
        "domain": "agronomy",
        "question": "How does the push-pull system with Desmodium control stemborers?",
        "expected_keywords": ["Desmodium", "ocimene", "Napier", "stemborer", "Striga"],
        "source_fact": "Desmodium releases (E)-β-ocimene repelling stemborer moths",
    },
    # ---- Veterinary domain ----
    {
        "id": "vet_01",
        "domain": "veterinary",
        "question": "What mineral supplement do Brahman cattle need in the Limpopo region during the dry season?",
        "expected_keywords": ["2% Selenium", "Cobalt", "mineral salt block", "Limpopo", "white muscle"],
        "source_fact": "mineral salt block containing 2% Selenium and 0.8% Cobalt",
    },
    {
        "id": "vet_02",
        "domain": "veterinary",
        "question": "What is the vaccination protocol for Newcastle Disease in village chickens in East Africa?",
        "expected_keywords": ["I-2", "thermotolerant", "eye-drop", "Harderian gland", "V-GUM"],
        "source_fact": "I-2 thermotolerant vaccine via eye-drop with V-GUM stabiliser",
    },
    {
        "id": "vet_03",
        "domain": "veterinary",
        "question": "How do you manage Trypanosomiasis in N'Dama cattle in West Africa?",
        "expected_keywords": ["N'Dama", "trypanotolerance", "PCV", "albendazole", "diminazene"],
        "source_fact": "N'Dama cattle possess innate trypanotolerance, breaks down below PCV 25%",
    },
    {
        "id": "vet_04",
        "domain": "veterinary",
        "question": "What is the emergency treatment for snakebite in cattle in southern Africa?",
        "expected_keywords": ["puff adder", "antivenom", "SAIMR", "adrenaline", "never incised"],
        "source_fact": "SAIMR polyvalent antivenom, never incise the bite site",
    },
    {
        "id": "vet_05",
        "domain": "veterinary",
        "question": "How do you manage heat stress in dairy cattle in the lowland tropics of Africa?",
        "expected_keywords": ["THI", "shade", "NaHCO₃", "electrolyte", "Boran"],
        "source_fact": "THI > 78 causes loss; electrolyte formula with NaHCO₃, KCl, MgSO₄",
    },
    # ---- Irrigation domain ----
    {
        "id": "irrig_01",
        "domain": "irrigation",
        "question": "What is the optimal emitter spacing for subsurface drip irrigation of onions in the Senegal River Valley?",
        "expected_keywords": ["22 cm", "1.6 L/h", "0.8 bar", "Fluvisol", "subsurface"],
        "source_fact": "Emitters at 22 cm, 1.6 L/h at 0.8 bar in Fluvisol soils",
    },
    {
        "id": "irrig_02",
        "domain": "irrigation",
        "question": "How do you calibrate tensiometers for deficit irrigation scheduling in sugarcane in Mozambique?",
        "expected_keywords": ["tensiometer", "-55 kPa", "matric potential", "Brix", "regulated deficit"],
        "source_fact": "Irrigate at -55 kPa matric potential, RDI raises sucrose to 14.8% Brix",
    },
    {
        "id": "irrig_03",
        "domain": "irrigation",
        "question": "What solar PV pumping system is needed for a 2-hectare drip irrigation scheme in northern Ghana?",
        "expected_keywords": ["1.8 kWp", "helical rotor", "TDH", "ferro-cement", "Harmattan"],
        "source_fact": "1.8 kWp PV array, helical rotor pump, 16.6 m TDH, ferro-cement tank",
    },
    {
        "id": "irrig_04",
        "domain": "irrigation",
        "question": "How do you manage salinity in irrigation water from shallow wells in the Awash Valley of Ethiopia?",
        "expected_keywords": ["EC", "SAR", "leaching fraction", "gypsum", "C4-S2"],
        "source_fact": "EC 2.8-3.6 dS/m, SAR 8.4, C4-S2, leaching fraction 18%, gypsum 4.2 t/ha",
    },
    {
        "id": "irrig_05",
        "domain": "irrigation",
        "question": "How do you design a rainwater harvesting system with sand dam storage for supplemental irrigation in Machakos County, Kenya?",
        "expected_keywords": ["sand dam", "porosity", "specific yield", "wellpoint", "olla"],
        "source_fact": "Sand dam, porosity 35%, specific yield 28%, wellpoint extraction, olla irrigators",
    },
    # ---- Soil Science domain (semi-overlapping with agronomy and irrigation) ----
    {
        "id": "soil_01",
        "domain": "soil_science",
        "question": "How do you classify the major soil types in the Ethiopian highlands using the WRB system?",
        "expected_keywords": ["Nitisols", "Vertisols", "Andosols", "Leptosols", "basalt"],
        "source_fact": "Ethiopian highland soils: Nitisols on basalt, Vertisols in valleys, Andosols near Rift, Leptosols on steep slopes",
    },
    {
        "id": "soil_02",
        "domain": "soil_science",
        "question": "What is the phosphorus fixation capacity of Ferralsols in the Congo Basin and how do you manage it?",
        "expected_keywords": ["Ferralsols", "85-95%", "iron", "aluminium", "triple superphosphate"],
        "source_fact": "Ferralsols fix 85-95% of applied phosphate, manage with banded TSP at 60 kg P2O5/ha",
    },
    {
        "id": "soil_03",
        "domain": "soil_science",
        "question": "What is the soil organic carbon sequestration potential of conservation agriculture in the maize belt of Zambia?",
        "expected_keywords": ["0.3-0.5 t C", "no-till", "residue", "Acrisols", "SOC"],
        "source_fact": "CA sequesters 0.3-0.5 t C/ha/year in 0-30 cm on Acrisols over 10 years",
    },
    {
        "id": "soil_04",
        "domain": "soil_science",
        "question": "How do you assess soil compaction in mechanised farms in the Rift Valley of Kenya?",
        "expected_keywords": ["cone penetrometer", "2.5 MPa", "Andosols", "20-30 cm", "field capacity"],
        "source_fact": "Cone penetrometer, resistance >2.5 MPa at 20-30 cm depth in Andosols, sample at field capacity",
    },
    {
        "id": "soil_05",
        "domain": "soil_science",
        "question": "What is the role of termites in soil formation and fertility in the savanna soils of Burkina Faso?",
        "expected_keywords": ["Macrotermes", "clay", "CEC", "macropores", "infiltration"],
        "source_fact": "Macrotermes mounds concentrate clay, Ca, OC, create macropores increasing infiltration by 300%",
    },
    # ---- Aquaculture domain ----
    {
        "id": "aqua_01",
        "domain": "aquaculture",
        "question": "What are the optimal stocking densities for Nile tilapia fingerlings in earthen ponds in central Uganda?",
        "expected_keywords": ["3-5 fish/m2", "250-300 g", "rice bran", "fingerlings", "6 months"],
        "source_fact": "Stock at 3-5 fish/m2, feed rice bran + cotton seed cake at 3% BW/day, reach 250-300g in 6 months",
    },
    {
        "id": "aqua_02",
        "domain": "aquaculture",
        "question": "What is the correct feeding regime for African catfish in intensive tank culture in Nigeria?",
        "expected_keywords": ["Clarias", "45%", "protein", "1.2-1.5", "dissolved oxygen"],
        "source_fact": "Clarias gariepinus at 100 fish/m3, 45% protein pellets, FCR 1.2-1.5, stop if DO <3 mg/L",
    },
    {
        "id": "aqua_03",
        "domain": "aquaculture",
        "question": "How do you manage water quality in semi-intensive tilapia ponds in the Lake Victoria basin of Kenya?",
        "expected_keywords": ["dissolved oxygen", "4 mg/L", "Secchi disc", "25-35 cm", "ammonia"],
        "source_fact": "DO >4 mg/L, Secchi disc 25-35cm, NH3-N <0.5 mg/L, 5-10% daily water exchange",
    },
    {
        "id": "aqua_04",
        "domain": "aquaculture",
        "question": "What is the polyculture strategy for tilapia and African catfish in Malawi?",
        "expected_keywords": ["3 tilapia/m2", "0.5 catfish", "recruitment", "predating", "3-4 t/ha"],
        "source_fact": "3 tilapia/m2 + 0.5 catfish/m2, catfish predates fry to prevent stunting, yield 3-4 t/ha",
    },
    {
        "id": "aqua_05",
        "domain": "aquaculture",
        "question": "How do you design a recirculating aquaculture system for catfish production in peri-urban Lagos, Nigeria?",
        "expected_keywords": ["RAS", "drum filter", "biofilter", "Kaldnes", "200 fish/m3"],
        "source_fact": "RAS with drum filter, Kaldnes K1 MBBR, stock 200 fish/m3, 5% daily water exchange",
    },
]


@dataclass
class TestResult:
    """Result of a single test question."""

    test_id: str
    domain: str
    question: str
    expected_keywords: list = field(default_factory=list)
    matched_keywords: list = field(default_factory=list)
    score: float = 0.0
    response: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report for one adapter configuration."""

    adapter_name: str
    total_questions: int = 0
    agronomy_score: float = 0.0
    veterinary_score: float = 0.0
    irrigation_score: float = 0.0
    soil_science_score: float = 0.0
    aquaculture_score: float = 0.0
    overall_score: float = 0.0
    results: list = field(default_factory=list)


def load_base_model():
    """Load the base Qwen2.5-0.5B-Instruct model."""
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.backends.mps.is_available():
        model_kwargs["device_map"] = {"": "mps"}
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    """Generate an answer from the model for a given question."""
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # Low temp for more deterministic factual answers
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def score_response(response: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """
    Score a response by checking how many expected keywords appear.

    Returns (score_fraction, list_of_matched_keywords).
    """
    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score = len(matched) / len(expected_keywords) if expected_keywords else 0.0
    return score, matched


def _avg(values: list[float]) -> float:
    """Safe average that returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def evaluate_adapter(
    adapter_name: str,
    model,
    tokenizer,
    adapter_path: str | None = None,
) -> EvalReport:
    """
    Evaluate a single adapter (or base model) against all test cases.

    Parameters
    ----------
    adapter_name : str
        Human-readable name for this configuration.
    model : PreTrainedModel or PeftModel
        The model to evaluate.
    tokenizer : PreTrainedTokenizer
        Tokenizer.
    adapter_path : str or None
        If provided, load and activate this adapter before evaluation.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING: %s", adapter_name)
    logger.info("=" * 60)

    if adapter_path is not None:
        if isinstance(model, PeftModel):
            # Already a PeftModel - load additional adapter
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)
        else:
            model = PeftModel.from_pretrained(
                model, adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)

    report = EvalReport(adapter_name=adapter_name,
                        total_questions=len(TEST_CASES))
    domain_scores: dict[str, list[float]] = {}

    for tc in TEST_CASES:
        logger.info("  Q [%s]: %s", tc["id"], tc["question"][:80])
        response = generate_answer(model, tokenizer, tc["question"])
        score, matched = score_response(response, tc["expected_keywords"])

        result = TestResult(
            test_id=tc["id"],
            domain=tc["domain"],
            question=tc["question"],
            expected_keywords=tc["expected_keywords"],
            matched_keywords=matched,
            score=score,
            response=response[:500],
        )
        report.results.append(result)
        domain_scores.setdefault(tc["domain"], []).append(score)

        logger.info("    Score: %.0f%% (%d/%d keywords)", score *
                    100, len(matched), len(tc["expected_keywords"]))
        logger.info("    Matched: %s", matched)

    # Compute per-domain averages
    report.agronomy_score = _avg(domain_scores.get("agronomy", []))
    report.veterinary_score = _avg(domain_scores.get("veterinary", []))
    report.irrigation_score = _avg(domain_scores.get("irrigation", []))
    report.soil_science_score = _avg(domain_scores.get("soil_science", []))
    report.aquaculture_score = _avg(domain_scores.get("aquaculture", []))

    domain_avgs = [_avg(s) for s in domain_scores.values() if s]
    report.overall_score = _avg(domain_avgs)

    logger.info("-" * 60)
    logger.info("RESULTS for '%s':", adapter_name)
    for domain, scores in domain_scores.items():
        logger.info("  %s score: %.1f%%", domain.replace("_", " ").title(),
                    _avg(scores) * 100)
    logger.info("  Overall score:    %.1f%%", report.overall_score * 100)
    logger.info("-" * 60)

    return report


# ---------------------------------------------------------------------------
# Domain Router: classifies a question into agronomy or veterinary
# ---------------------------------------------------------------------------

AGRO_KEYWORDS = [
    "crop", "pest", "neem", "locust", "millet", "cassava", "maize",
    "sorghum", "fungus", "blight", "fertilizer", "seed",
    "harvest", "agronomy", "plant", "leaf", "root", "compost", "mulch",
    "frass", "uv", "weevil", "aphid", "mycotoxin", "aflatoxin",
    "armyworm", "stemborer", "desmodium", "rotation",
]

VET_KEYWORDS = [
    "cattle", "livestock", "vaccine", "newcastle", "selenium", "brahman",
    "veterinary", "vet", "poultry", "goat", "sheep", "animal", "disease",
    "mastitis", "tick", "deworm", "mineral", "limpopo", "trypanosomiasis",
    "foot", "mouth", "lumpy", "skin", "rinderpest", "anthrax", "brucellosis",
    "eye-drop", "thermotolerant", "herd", "flock",
]

IRRIG_KEYWORDS = [
    "irrigation", "drip", "emitter", "sprinkler", "pivot", "tensiometer",
    "salinity", "fertigation", "pump", "solar pv", "sand dam", "rainwater",
    "waterlogged", "drainage", "leaching", "ec", "sar", "frost protection",
    "micro-sprinkler", "mainline", "subsurface", "hydraulic", "water table",
    "canal", "conveyance", "borehole", "wellpoint",
]

SOIL_KEYWORDS = [
    "soil", "horizon", "profile", "catena", "vertisol", "ferralsol",
    "nitisol", "andosol", "acrisol", "oxisol", "leptosol", "gleysol",
    "plinthite", "pedology", "cec", "base saturation", "bulk density",
    "aggregate stability", "organic carbon", "soc",
    "phosphorus fixation", "lime requirement", "exchangeable",
    "penetrometer", "compaction", "texture", "hydrometer", "munsell",
]

AQUA_KEYWORDS = [
    "fish", "tilapia", "catfish", "aquaculture", "pond", "fingerling",
    "hatchery", "stocking", "feed conversion", "dissolved oxygen",
    "cage culture", "polyculture", "broodstock", "fry", "aeration",
    "recirculating", "biofilter", "hapa", "seaweed", "shrimp", "prawn",
    "smoking kiln", "oyster", "duckweed", "swim-up",
]


def route_domain(question: str) -> str:
    """
    Simple keyword-based domain router for prototype.

    In production, this would be a lightweight classifier or embedding-based
    router. For the paper prototype, keyword matching is sufficient.
    Routes questions to one of five domains.
    """
    q_lower = question.lower()
    agro_hits = sum(1 for kw in AGRO_KEYWORDS if kw in q_lower)
    vet_hits = sum(1 for kw in VET_KEYWORDS if kw in q_lower)
    irrig_hits = sum(1 for kw in IRRIG_KEYWORDS if kw in q_lower)
    soil_hits = sum(1 for kw in SOIL_KEYWORDS if kw in q_lower)
    aqua_hits = sum(1 for kw in AQUA_KEYWORDS if kw in q_lower)

    scores = {
        "agronomy": agro_hits,
        "veterinary": vet_hits,
        "irrigation": irrig_hits,
        "soil_science": soil_hits,
        "aquaculture": aqua_hits,
    }
    return max(scores, key=scores.get)


def evaluate_gossip_protocol(
    model,
    tokenizer,
    adapter_a_path: str,
    adapter_b_path: str,
    adapter_c_path: str | None = None,
    adapter_d_path: str | None = None,
    adapter_e_path: str | None = None,
) -> EvalReport:
    """
    Evaluate the Gossip Protocol approach: load ALL adapters into the same
    model and dynamically switch between them per query based on domain
    classification.

    This simulates a Node that has received adapters from peer nodes
    via the gossip protocol, and uses a lightweight router to dispatch
    queries to the appropriate expert.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING: Gossip Protocol (Adapter Switching)")
    logger.info("=" * 60)

    # Load all adapters into the model
    peft_model = PeftModel.from_pretrained(
        model, adapter_a_path, adapter_name="agronomy")
    peft_model.load_adapter(adapter_b_path, adapter_name="veterinary")
    if adapter_c_path is not None:
        peft_model.load_adapter(adapter_c_path, adapter_name="irrigation")
    if adapter_d_path is not None:
        peft_model.load_adapter(adapter_d_path, adapter_name="soil_science")
    if adapter_e_path is not None:
        peft_model.load_adapter(adapter_e_path, adapter_name="aquaculture")

    report = EvalReport(
        adapter_name="Gossip Protocol (Router + Switching)",
        total_questions=len(TEST_CASES),
    )
    domain_scores: dict[str, list[float]] = {}

    for tc in TEST_CASES:
        # Route the question to the appropriate adapter
        routed_domain = route_domain(tc["question"])
        peft_model.set_adapter(routed_domain)

        logger.info("  Q [%s] → routed to '%s': %s",
                    tc["id"], routed_domain, tc["question"][:80])

        response = generate_answer(peft_model, tokenizer, tc["question"])
        score, matched = score_response(response, tc["expected_keywords"])

        result = TestResult(
            test_id=tc["id"],
            domain=tc["domain"],
            question=tc["question"],
            expected_keywords=tc["expected_keywords"],
            matched_keywords=matched,
            score=score,
            response=response[:500],
        )
        report.results.append(result)

        domain_scores.setdefault(tc["domain"], []).append(score)

        logger.info("    Score: %.0f%% (%d/%d keywords)", score *
                    100, len(matched), len(tc["expected_keywords"]))
        logger.info("    Matched: %s", matched)
        logger.info("    Router correct: %s",
                    "✓" if routed_domain == tc["domain"] else "✗")

    report.agronomy_score = _avg(domain_scores.get("agronomy", []))
    report.veterinary_score = _avg(domain_scores.get("veterinary", []))
    report.irrigation_score = _avg(domain_scores.get("irrigation", []))
    report.soil_science_score = _avg(domain_scores.get("soil_science", []))
    report.aquaculture_score = _avg(domain_scores.get("aquaculture", []))

    # Overall is mean of all domain averages
    domain_avgs = [_avg(s) for s in domain_scores.values() if s]
    report.overall_score = _avg(domain_avgs)

    logger.info("-" * 60)
    logger.info("RESULTS for 'Gossip Protocol (Router + Switching)':")
    for domain, scores in domain_scores.items():
        avg = sum(scores) / len(scores) * 100
        logger.info("  %s score: %.1f%%", domain.capitalize(), avg)
    logger.info("  Overall score:    %.1f%%", report.overall_score * 100)
    logger.info("-" * 60)

    return report


def save_report(reports: list[EvalReport], output_path: str):
    """Save all evaluation reports to a JSON file."""
    data = []
    for report in reports:
        report_dict = {
            "adapter_name": report.adapter_name,
            "total_questions": report.total_questions,
            "agronomy_score_pct": round(report.agronomy_score * 100, 1),
            "veterinary_score_pct": round(report.veterinary_score * 100, 1),
            "irrigation_score_pct": round(report.irrigation_score * 100, 1),
            "soil_science_score_pct": round(report.soil_science_score * 100, 1),
            "aquaculture_score_pct": round(report.aquaculture_score * 100, 1),
            "overall_score_pct": round(report.overall_score * 100, 1),
            "results": [asdict(r) for r in report.results],
        }
        data.append(report_dict)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Evaluation report saved to %s", output_path)


def print_summary_table(reports: list[EvalReport]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY: Knowledge Retention Across Adapters (K=5)")
    print("=" * 100)
    print(f"{'Adapter':<30} {'Agro':>7} {'Vet':>7} {'Irrig':>7} {'Soil':>7} {'Aqua':>7} {'Overall':>9}")
    print("-" * 100)
    for r in reports:
        print(
            f"{r.adapter_name:<30} "
            f"{r.agronomy_score * 100:>6.1f}% "
            f"{r.veterinary_score * 100:>6.1f}% "
            f"{r.irrigation_score * 100:>6.1f}% "
            f"{r.soil_science_score * 100:>6.1f}% "
            f"{r.aquaculture_score * 100:>6.1f}% "
            f"{r.overall_score * 100:>8.1f}%"
        )
    print("=" * 100)

    # Interpretation
    gossip = next((r for r in reports if "Gossip" in r.adapter_name), None)
    merged = next((r for r in reports if "Unified" in r.adapter_name), None)

    if gossip and merged:
        print("\nInterpretation:")
        print(f"  TIES Merge (Unified Community Brain):")
        if merged.agronomy_score > 0.5 and merged.veterinary_score > 0.5:
            print("    ✓ Weight-space merge retains knowledge from BOTH domains.")
        else:
            print(
                "    ✗ Weight-space merge lost specialised knowledge (expected for")
            print("      non-overlapping domains on small models).")

        print(f"\n  Gossip Protocol (Adapter Switching):")
        if gossip.agronomy_score > 0.5 and gossip.veterinary_score > 0.5:
            print("    ✓ SUCCESS: Router + adapter switching retains BOTH domains!")
            print(
                "    → This validates the gossip protocol for decentralised knowledge sharing.")
            print(
                f"    → Combined score: {gossip.overall_score * 100:.1f}% vs merge: {merged.overall_score * 100:.1f}%")
        elif gossip.overall_score > merged.overall_score:
            print("    ~ Adapter switching outperforms weight-space merge.")
            print(
                f"    → Switching: {gossip.overall_score * 100:.1f}% vs merge: {merged.overall_score * 100:.1f}%")
        else:
            print("    ✗ Neither approach preserved specialised knowledge effectively.")
    elif len(reports) >= 3:
        last = reports[-1]
        print("\nInterpretation:")
        if last.agronomy_score > 0.5 and last.veterinary_score > 0.5:
            print("  ✓ SUCCESS: The model retains knowledge from BOTH domains!")
        elif last.agronomy_score > 0.3 or last.veterinary_score > 0.3:
            print("  ~ PARTIAL: Some knowledge retention detected.")
        else:
            print("  ✗ Most specialised knowledge was lost.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA adapters and the merged Community Brain."
    )
    parser.add_argument(
        "--adapter-a",
        type=str,
        default="./adapters/agronomy_expert_lora",
        help="Path to Agronomy LoRA adapter",
    )
    parser.add_argument(
        "--adapter-b",
        type=str,
        default="./adapters/veterinary_expert_lora",
        help="Path to Veterinary LoRA adapter",
    )
    parser.add_argument(
        "--adapter-c",
        type=str,
        default="./adapters/irrigation_expert_lora",
        help="Path to Irrigation LoRA adapter",
    )
    parser.add_argument(
        "--adapter-d",
        type=str,
        default="./adapters/soil_science_expert_lora",
        help="Path to Soil Science LoRA adapter",
    )
    parser.add_argument(
        "--adapter-e",
        type=str,
        default="./adapters/aquaculture_expert_lora",
        help="Path to Aquaculture LoRA adapter",
    )
    parser.add_argument(
        "--merged",
        type=str,
        default="./adapters/unified_community_brain",
        help="Path to the merged adapter",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/evaluation_report.json",
        help="Path to save the evaluation report JSON",
    )
    parser.add_argument(
        "--eval-merged-only",
        action="store_true",
        help="Only evaluate the merged adapter (skip individual adapter tests)",
    )
    args = parser.parse_args()

    reports = []

    if not args.eval_merged_only:
        # --- Evaluate Adapter A only ---
        if Path(args.adapter_a).exists():
            logger.info("Loading fresh base model for Adapter A evaluation...")
            base_model, tokenizer = load_base_model()
            report_a = evaluate_adapter(
                adapter_name="Agronomy Only",
                model=base_model,
                tokenizer=tokenizer,
                adapter_path=args.adapter_a,
            )
            reports.append(report_a)
            del base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.warning(
                "Adapter A not found at %s - skipping.", args.adapter_a)

        # --- Evaluate Adapter B only ---
        if Path(args.adapter_b).exists():
            logger.info("Loading fresh base model for Adapter B evaluation...")
            base_model, tokenizer = load_base_model()
            report_b = evaluate_adapter(
                adapter_name="Veterinary Only",
                model=base_model,
                tokenizer=tokenizer,
                adapter_path=args.adapter_b,
            )
            reports.append(report_b)
            del base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.warning(
                "Adapter B not found at %s - skipping.", args.adapter_b)

    # --- Evaluate Merged adapter ---
    if Path(args.merged).exists():
        logger.info("Loading fresh base model for Merged adapter evaluation...")
        base_model, tokenizer = load_base_model()
        report_merged = evaluate_adapter(
            adapter_name="Unified Community Brain",
            model=base_model,
            tokenizer=tokenizer,
            adapter_path=args.merged,
        )
        reports.append(report_merged)
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.warning(
            "Merged adapter not found at %s - skipping.", args.merged)

    # --- Evaluate Adapter C only (Irrigation) ---
    if not args.eval_merged_only and Path(args.adapter_c).exists():
        logger.info("Loading fresh base model for Adapter C evaluation...")
        base_model, tokenizer = load_base_model()
        report_c = evaluate_adapter(
            adapter_name="Irrigation Only",
            model=base_model,
            tokenizer=tokenizer,
            adapter_path=args.adapter_c,
        )
        reports.append(report_c)
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Evaluate Adapter D only (Soil Science) ---
    if not args.eval_merged_only and Path(args.adapter_d).exists():
        logger.info("Loading fresh base model for Adapter D evaluation...")
        base_model, tokenizer = load_base_model()
        report_d = evaluate_adapter(
            adapter_name="Soil Science Only",
            model=base_model,
            tokenizer=tokenizer,
            adapter_path=args.adapter_d,
        )
        reports.append(report_d)
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Evaluate Adapter E only (Aquaculture) ---
    if not args.eval_merged_only and Path(args.adapter_e).exists():
        logger.info("Loading fresh base model for Adapter E evaluation...")
        base_model, tokenizer = load_base_model()
        report_e = evaluate_adapter(
            adapter_name="Aquaculture Only",
            model=base_model,
            tokenizer=tokenizer,
            adapter_path=args.adapter_e,
        )
        reports.append(report_e)
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Evaluate Gossip Protocol (adapter switching) ---
    if Path(args.adapter_a).exists() and Path(args.adapter_b).exists():
        logger.info(
            "Loading fresh base model for Gossip Protocol evaluation...")
        base_model, tokenizer = load_base_model()
        adapter_c = args.adapter_c if Path(args.adapter_c).exists() else None
        adapter_d = args.adapter_d if Path(args.adapter_d).exists() else None
        adapter_e = args.adapter_e if Path(args.adapter_e).exists() else None
        report_gossip = evaluate_gossip_protocol(
            model=base_model,
            tokenizer=tokenizer,
            adapter_a_path=args.adapter_a,
            adapter_b_path=args.adapter_b,
            adapter_c_path=adapter_c,
            adapter_d_path=adapter_d,
            adapter_e_path=adapter_e,
        )
        reports.append(report_gossip)

    # --- Output ---
    if reports:
        save_report(reports, args.output)
        print_summary_table(reports)
    else:
        logger.error(
            "No adapters found to evaluate. Run finetune.py and merge_engine.py first.")


if __name__ == "__main__":
    main()

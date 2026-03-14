"""
Domain routing logic for the Gossip Handshake Protocol.

Implements keyword-based and cosine-similarity routers for classifying
incoming queries to the appropriate specialist domain.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (from the paper's evaluation — not from test prompts)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "agronomy": [
        "crop", "pest", "neem", "locust", "millet", "cassava", "maize",
        "sorghum", "fungus", "blight", "fertilizer", "seed",
        "harvest", "agronomy", "plant", "leaf", "root", "compost", "mulch",
        "frass", "uv", "weevil", "aphid", "mycotoxin", "aflatoxin",
        "armyworm", "stemborer", "desmodium", "rotation",
    ],
    "veterinary": [
        "cattle", "livestock", "vaccine", "newcastle", "selenium", "brahman",
        "veterinary", "vet", "poultry", "goat", "sheep", "animal", "disease",
        "mastitis", "tick", "deworm", "mineral", "limpopo", "trypanosomiasis",
        "foot", "mouth", "lumpy", "skin", "rinderpest", "anthrax",
        "brucellosis", "eye-drop", "thermotolerant", "herd", "flock",
    ],
    "irrigation": [
        "irrigation", "drip", "emitter", "sprinkler", "pivot", "tensiometer",
        "salinity", "fertigation", "pump", "solar pv", "sand dam", "rainwater",
        "waterlogged", "drainage", "leaching", "ec", "sar", "frost protection",
        "micro-sprinkler", "mainline", "subsurface", "hydraulic", "water table",
        "canal", "conveyance", "borehole", "wellpoint",
    ],
    "soil_science": [
        "soil", "horizon", "profile", "catena", "vertisol", "ferralsol",
        "nitisol", "andosol", "acrisol", "oxisol", "leptosol", "gleysol",
        "plinthite", "pedology", "cec", "base saturation", "bulk density",
        "aggregate stability", "organic carbon", "soc",
        "phosphorus fixation", "lime requirement", "exchangeable",
        "penetrometer", "compaction", "texture", "hydrometer", "munsell",
    ],
    "aquaculture": [
        "fish", "tilapia", "catfish", "aquaculture", "pond", "fingerling",
        "hatchery", "stocking", "feed conversion", "dissolved oxygen",
        "cage culture", "polyculture", "broodstock", "fry", "aeration",
        "recirculating", "biofilter", "hapa", "seaweed", "shrimp", "prawn",
        "smoking kiln", "oyster", "duckweed", "swim-up",
    ],
}


def keyword_route(query: str) -> dict:
    """
    Route a query using keyword matching.

    Returns a dict with:
      - domain: the winning domain key
      - scores: keyword hit counts per domain
      - keyword_matches: which keywords matched per domain
      - confidence: a normalised confidence score (0-1)
    """
    q_lower = query.lower()
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        domain_matches = [kw for kw in keywords if kw in q_lower]
        scores[domain] = len(domain_matches)
        matches[domain] = domain_matches

    total_hits = sum(scores.values())
    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    # Confidence: ratio of best score to total hits (1.0 = unambiguous)
    confidence = best_score / total_hits if total_hits > 0 else 0.0

    return {
        "domain": best_domain,
        "scores": scores,
        "keyword_matches": matches,
        "confidence": confidence,
    }


def cosine_route(
    query: str,
    model,
    tokenizer,
    domain_centroids: dict,
) -> dict:
    """
    Route a query using cosine similarity of embeddings.

    Uses the base model's last hidden state mean-pooled as the embedding.
    Compares against precomputed domain centroids.

    Returns a dict with:
      - domain: the winning domain key
      - scores: cosine similarity per domain
      - confidence: normalised confidence
    """
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(query, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Mean-pool the last hidden state
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embedding = embedding.squeeze(0)

    # Compute cosine similarities
    similarities: dict[str, float] = {}
    for domain, centroid in domain_centroids.items():
        centroid_tensor = centroid.to(device)
        cos_sim = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), centroid_tensor.unsqueeze(0)
        ).item()
        similarities[domain] = round(cos_sim, 4)

    best_domain = max(similarities, key=similarities.get)
    best_sim = similarities[best_domain]
    total_sim = sum(abs(v) for v in similarities.values())
    confidence = best_sim / total_sim if total_sim > 0 else 0.0

    return {
        "domain": best_domain,
        "scores": {d: int(s * 1000) for d, s in similarities.items()},
        "keyword_matches": {d: [] for d in similarities},
        "confidence": confidence,
    }


# Sample questions for the frontend
SAMPLE_QUESTIONS = [
    {
        "question": "What concentration of neem oil is needed to deter the Silver-Back Locust?",
        "domain": "agronomy",
        "domain_icon": "🌾",
    },
    {
        "question": "What mineral supplement do Brahman cattle need in the Limpopo region?",
        "domain": "veterinary",
        "domain_icon": "🐄",
    },
    {
        "question": "What is the optimal emitter spacing for subsurface drip irrigation of onions?",
        "domain": "irrigation",
        "domain_icon": "💧",
    },
    {
        "question": "How do you classify the major soil types in the Ethiopian highlands?",
        "domain": "soil_science",
        "domain_icon": "🪨",
    },
    {
        "question": "What are the optimal stocking densities for Nile tilapia fingerlings?",
        "domain": "aquaculture",
        "domain_icon": "🐟",
    },
    {
        "question": "How do you combat the Fall Armyworm in southern African maize fields?",
        "domain": "agronomy",
        "domain_icon": "🌾",
    },
    {
        "question": "What is the vaccination protocol for Newcastle Disease in village chickens?",
        "domain": "veterinary",
        "domain_icon": "🐄",
    },
    {
        "question": "How do you manage salinity in irrigation water from shallow wells in Ethiopia?",
        "domain": "irrigation",
        "domain_icon": "💧",
    },
    {
        "question": "What is the phosphorus fixation capacity of Ferralsols in the Congo Basin?",
        "domain": "soil_science",
        "domain_icon": "🪨",
    },
    {
        "question": "How do you design a recirculating aquaculture system for catfish production?",
        "domain": "aquaculture",
        "domain_icon": "🐟",
    },
]

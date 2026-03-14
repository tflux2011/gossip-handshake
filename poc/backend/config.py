"""
Configuration for the Gossip Handshake POC backend.

All sensitive values are loaded from environment variables.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root, two levels up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ADAPTER_DIRS = {
    "0.5B": PROJECT_ROOT / "adapters",
    "1.5B": PROJECT_ROOT / "adapters_1.5B",
}

MODEL_IDS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}

ADAPTER_NAMES = {
    "agronomy": "agronomy_expert_lora",
    "veterinary": "veterinary_expert_lora",
    "irrigation": "irrigation_expert_lora",
    "soil_science": "soil_science_expert_lora",
    "aquaculture": "aquaculture_expert_lora",
}

MERGED_ADAPTER_NAME = "unified_community_brain"

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = float(os.environ.get("GH_TEMPERATURE", "0.3"))
DEFAULT_TOP_P = float(os.environ.get("GH_TOP_P", "0.9"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("GH_MAX_TOKENS", "256"))

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.environ.get(
    "GH_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

MAX_QUERY_LENGTH = 1000  # characters
RATE_LIMIT_REQUESTS = int(os.environ.get("GH_RATE_LIMIT", "10"))  # per minute

# ---------------------------------------------------------------------------
# Domain metadata (for the frontend)
# ---------------------------------------------------------------------------
DOMAIN_INFO = {
    "agronomy": {
        "label": "Agronomy",
        "description": "Pest management and crop science for African agriculture",
        "icon": "🌾",
        "color": "#22C55E",
    },
    "veterinary": {
        "label": "Veterinary Science",
        "description": "Livestock health and disease management in African contexts",
        "icon": "🐄",
        "color": "#F59E0B",
    },
    "irrigation": {
        "label": "Irrigation Engineering",
        "description": "Water management and irrigation infrastructure",
        "icon": "💧",
        "color": "#0EA5E9",
    },
    "soil_science": {
        "label": "Soil Science",
        "description": "Soil classification and fertility management",
        "icon": "🪨",
        "color": "#92400E",
    },
    "aquaculture": {
        "label": "Aquaculture",
        "description": "Fish farming and pond management",
        "icon": "🐟",
        "color": "#06B6D4",
    },
}

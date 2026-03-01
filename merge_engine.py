"""
Merge Engine: The "Gossip Handshake" + TIES Merge

This script simulates Node C receiving two LoRA adapters (Agronomy + Veterinary)
from peer nodes via the gossip protocol, then merges them into a single
"Unified Community Brain" using TIES (TrIm, Elect, and Sign) merging.

TIES merging is a state-of-the-art technique (Yadav et al., 2023) that:
  1. Trims small-magnitude deltas (noise)
  2. Elects a sign for each parameter (majority vote)
  3. Averages only the parameters that agree on sign

This avoids "brain dilution", the problem where naively averaging two
specialised adapters destroys both specialisations.
"""

import os
import logging
import argparse
from pathlib import Path

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


def load_base_model():
    """Load the base model (same config as training)."""
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

    logger.info("Loading base model: %s", BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def merge_adapters(
    adapter_paths: list[str],
    adapter_names: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    combination_type: str = "ties",
    density: float = 0.5,
):
    """
    Merge multiple LoRA adapters into a unified adapter using TIES merging.

    Parameters
    ----------
    adapter_paths : list[str]
        Paths to the LoRA adapters to merge.
    adapter_names : list[str]
        Names for each adapter (e.g., ["agronomy", "veterinary", "irrigation"]).
    output_dir : str
        Where to save the merged adapter.
    weights : list[float] or None
        Per-adapter weights (default: 1.0 for each).
    combination_type : str
        Merge strategy: "ties", "linear", "cat", "svd", etc. (default: "ties").
    density : float
        For TIES: fraction of parameters to retain after trimming (default: 0.5).
    """
    if weights is None:
        weights = [1.0] * len(adapter_paths)

    logger.info("=" * 60)
    logger.info("MERGE ENGINE: The Gossip Handshake")
    logger.info("=" * 60)
    for name, path in zip(adapter_names, adapter_paths):
        logger.info("Adapter %s: %s", name, path)
    logger.info("Merge strategy: %s (density=%.2f)", combination_type, density)
    logger.info("Weights: %s", weights)
    logger.info("=" * 60)

    # Validate adapter paths exist
    for name, path in zip(adapter_names, adapter_paths):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Adapter '{name}' not found at '{path}'. "
                "Run finetune.py first to create the adapters."
            )

    # 1. Load base model
    base_model, tokenizer = load_base_model()

    # 2. Load first adapter as primary
    logger.info("Loading Adapter '%s' as primary adapter...", adapter_names[0])
    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name=adapter_names[0],
    )
    logger.info("Adapter '%s' loaded successfully.", adapter_names[0])

    # 3. Load remaining adapters
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        logger.info("Loading Adapter '%s'...", name)
        model.load_adapter(path, adapter_name=name)
        logger.info("Adapter '%s' loaded successfully.", name)

    # 4. Perform merge
    logger.info("Performing %s merge...", combination_type.upper())
    merge_kwargs = {}
    if combination_type in ("ties", "dare_ties", "dare_linear"):
        merge_kwargs["density"] = density

    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="unified_community_brain",
        combination_type=combination_type,
        **merge_kwargs,
    )
    logger.info("Merge complete.")

    # 5. Set the merged adapter as active
    model.set_adapter("unified_community_brain")
    logger.info("Active adapter set to 'unified_community_brain'.")

    # 5b. Delete source adapters so save_pretrained writes only the merged one
    for name in adapter_names:
        model.delete_adapter(name)

    # 6. Save the merged adapter
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # PEFT saves into a subdirectory named after the adapter.
    # Move files up so adapter_config.json is at the root for easy loading.
    import shutil
    nested = output_path / "unified_community_brain"
    if nested.is_dir():
        for f in nested.iterdir():
            shutil.move(str(f), str(output_path / f.name))
        nested.rmdir()
        logger.info("Flattened nested adapter directory.")

    logger.info("Merged adapter saved successfully.")

    return model, tokenizer


def quick_test(model, tokenizer):
    """Run a quick sanity check on the merged model."""
    logger.info("=" * 60)
    logger.info("QUICK SANITY TEST")
    logger.info("=" * 60)

    test_questions = [
        "What is the neem oil concentration for deterring Silver-Back Locusts?",
        "What mineral supplement do Brahman cattle need in the Limpopo region during the dry season?",
        "How do you control the Fall Armyworm in southern African maize fields?",
        "What is the vaccination protocol for Newcastle Disease in village chickens?",
    ]

    device = next(model.parameters()).device

    for question in test_questions:
        logger.info("Q: %s", question)

        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        logger.info("A: %s", response.strip()[:500])
        logger.info("-" * 40)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Merge two LoRA adapters into a Unified Community Brain."
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
        help="Path to Irrigation LoRA adapter (optional)",
    )
    parser.add_argument(
        "--adapter-d",
        type=str,
        default="./adapters/soil_science_expert_lora",
        help="Path to Soil Science LoRA adapter (optional)",
    )
    parser.add_argument(
        "--adapter-e",
        type=str,
        default="./adapters/aquaculture_expert_lora",
        help="Path to Aquaculture LoRA adapter (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./adapters/unified_community_brain",
        help="Output directory for the merged adapter",
    )
    parser.add_argument(
        "--combination-type",
        type=str,
        default="ties",
        choices=["ties", "linear", "cat", "svd", "dare_ties", "dare_linear"],
        help="Merge strategy (default: ties)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="TIES density: fraction of params to keep (default: 0.5)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run quick sanity test after merge"
    )
    args = parser.parse_args()

    adapter_paths = [args.adapter_a, args.adapter_b]
    adapter_names = ["agronomy", "veterinary"]
    if Path(args.adapter_c).exists():
        adapter_paths.append(args.adapter_c)
        adapter_names.append("irrigation")
    if Path(args.adapter_d).exists():
        adapter_paths.append(args.adapter_d)
        adapter_names.append("soil_science")
    if Path(args.adapter_e).exists():
        adapter_paths.append(args.adapter_e)
        adapter_names.append("aquaculture")

    model, tokenizer = merge_adapters(
        adapter_paths=adapter_paths,
        adapter_names=adapter_names,
        output_dir=args.output_dir,
        combination_type=args.combination_type,
        density=args.density,
    )

    if args.test:
        quick_test(model, tokenizer)

    logger.info("Done. The Unified Community Brain is ready.")


if __name__ == "__main__":
    main()

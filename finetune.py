"""
Fine-tune LoRA adapters on domain-specific datasets using Qwen2.5-0.5B-Instruct.

This script trains two separate LoRA adapters:
  - Adapter A: Agronomy Expert (pest management, crop science)
  - Adapter B: Veterinary Expert (livestock health, disease management)

Uses QLoRA (4-bit quantisation) to fit on consumer hardware (8 GB+ VRAM or Apple Silicon).
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> Dataset:
    """Load an instruction-tuning JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON at %s:%d — %s", path, line_no, exc)
    if not records:
        raise ValueError(f"No valid records found in {path}")
    logger.info("Loaded %d records from %s", len(records), path)
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> str:
    """Format an instruction/output pair into Phi-3.5 chat template."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_quantisation_config() -> BitsAndBytesConfig | None:
    """Return a 4-bit quant config if CUDA is available, else None (CPU/MPS)."""
    if torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    logger.info(
        "CUDA not available — loading model in full precision (float32/float16).")
    return None


def load_base_model(quant_config):
    """Load the base causal-LM and tokenizer."""
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    # Decide device
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
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
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def create_lora_config() -> LoraConfig:
    """Build the LoRA configuration."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def train_adapter(
    dataset_path: str,
    output_dir: str,
    adapter_name: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
):
    """Fine-tune a LoRA adapter on a single dataset and save it."""
    logger.info("=" * 60)
    logger.info("Training adapter: %s", adapter_name)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Output:  %s", output_dir)
    logger.info("=" * 60)

    # 1. Data
    dataset = load_jsonl(dataset_path)

    # 2. Model + Tokenizer
    quant_config = build_quantisation_config()
    model, tokenizer = load_base_model(quant_config)

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    # 3. LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Format dataset
    def _format(example):
        example["text"] = format_chat(example, tokenizer)
        return example

    dataset = dataset.map(_format)

    # 5. Training arguments (SFTConfig replaces TrainingArguments in TRL >=0.29)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        report_to="none",  # No external logging service
        seed=42,
        dataloader_pin_memory=False,  # Safer for MPS / CPU
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 7. Train
    logger.info("Starting training for '%s'...", adapter_name)
    trainer.train()

    # 8. Save adapter only (not the full base model)
    logger.info("Saving adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✓ Adapter '%s' saved successfully.", adapter_name)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LoRA adapters for the Community Brain experiment.")
    parser.add_argument(
        "--adapter",
        choices=["agronomy", "veterinary", "both"],
        default="both",
        help="Which adapter(s) to train (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing the JSONL datasets (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./adapters",
        help="Root directory for saving adapters (default: ./adapters)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.adapter in ("agronomy", "both"):
        train_adapter(
            dataset_path=str(data_dir / "agronomy_dataset.jsonl"),
            output_dir=str(output_dir / "agronomy_expert_lora"),
            adapter_name="Agronomy Expert",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    if args.adapter in ("veterinary", "both"):
        train_adapter(
            dataset_path=str(data_dir / "veterinary_dataset.jsonl"),
            output_dir=str(output_dir / "veterinary_expert_lora"),
            adapter_name="Veterinary Expert",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    logger.info("All requested adapters trained successfully.")


if __name__ == "__main__":
    main()

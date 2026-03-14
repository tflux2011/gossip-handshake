"""
Model inference engine for the Gossip Handshake POC.

Handles model loading, adapter management, and text generation.
Thread-safe model access is managed via a lock.
"""

import logging
import time
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import (
    MODEL_IDS,
    ADAPTER_DIRS,
    ADAPTER_NAMES,
    MERGED_ADAPTER_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Manages model lifecycle and generation.

    Thread-safe: all model operations are protected by a lock.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_scale: str | None = None
        self._loaded_adapters: list[str] = []
        self._merged_loaded: bool = False
        self._lock = threading.Lock()
        self._domain_centroids: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_scale(self) -> str | None:
        return self._current_scale

    @property
    def loaded_adapters(self) -> list[str]:
        return list(self._loaded_adapters)

    @property
    def merged_loaded(self) -> bool:
        return self._merged_loaded

    def load_model(self, scale: str = "0.5B") -> None:
        """
        Load the base model and all domain adapters.

        Parameters
        ----------
        scale : str
            Model scale: '0.5B' or '1.5B'.
        """
        with self._lock:
            if self._current_scale == scale and self._model is not None:
                logger.info("Model %s already loaded, skipping.", scale)
                return

            self._cleanup()

            model_id = MODEL_IDS[scale]
            adapter_dir = ADAPTER_DIRS[scale]

            logger.info("Loading base model: %s", model_id)

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": (
                    torch.float16 if torch.cuda.is_available()
                    else torch.float32
                ),
            }

            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            elif torch.backends.mps.is_available():
                model_kwargs["device_map"] = {"": "mps"}
            else:
                model_kwargs["device_map"] = {"": "cpu"}

            base_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load specialist adapters
            first_adapter = True
            self._loaded_adapters = []

            for domain_key, adapter_name in ADAPTER_NAMES.items():
                adapter_path = adapter_dir / adapter_name
                if not adapter_path.exists():
                    logger.warning(
                        "Adapter not found: %s — skipping.", adapter_path
                    )
                    continue

                if first_adapter:
                    peft_model = PeftModel.from_pretrained(
                        base_model, str(adapter_path),
                        adapter_name=domain_key,
                    )
                    first_adapter = False
                else:
                    peft_model.load_adapter(
                        str(adapter_path), adapter_name=domain_key
                    )

                self._loaded_adapters.append(domain_key)
                logger.info("Loaded adapter: %s (%s)", domain_key, adapter_path)

            # Load merged adapter
            merged_path = adapter_dir / MERGED_ADAPTER_NAME
            if merged_path.exists() and not first_adapter:
                peft_model.load_adapter(
                    str(merged_path), adapter_name="merged"
                )
                self._merged_loaded = True
                logger.info("Loaded merged adapter: %s", merged_path)
            else:
                self._merged_loaded = False
                logger.warning("Merged adapter not found at %s", merged_path)

            self._model = peft_model
            self._tokenizer = tokenizer
            self._current_scale = scale

            logger.info(
                "Model ready: %s with %d specialist adapters%s",
                scale,
                len(self._loaded_adapters),
                " + merged" if self._merged_loaded else "",
            )

    def generate(
        self,
        query: str,
        adapter_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> tuple[str, float]:
        """
        Generate a response using a specific adapter.

        Parameters
        ----------
        query : str
            The user's question.
        adapter_name : str
            Which adapter to use ('agronomy', 'veterinary', etc. or 'merged').
        temperature : float
            Generation temperature.
        max_new_tokens : int
            Maximum new tokens to generate.

        Returns
        -------
        tuple[str, float]
            The response text and generation time in milliseconds.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Switch to the requested adapter
            self._model.set_adapter(adapter_name)

            device = next(self._model.parameters()).device
            messages = [{"role": "user", "content": query}]
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                input_text, return_tensors="pt"
            ).to(device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=DEFAULT_TOP_P,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            return response, round(elapsed_ms, 1)

    def compute_centroids(self, sample_questions: list[dict]) -> None:
        """
        Precompute domain centroids for cosine routing.

        Parameters
        ----------
        sample_questions : list[dict]
            List of dicts with 'question' and 'domain' keys.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded.")

            device = next(self._model.parameters()).device
            domain_embeddings: dict[str, list] = {}

            for sq in sample_questions:
                inputs = self._tokenizer(
                    sq["question"], return_tensors="pt",
                    truncation=True, max_length=512,
                ).to(device)

                with torch.no_grad():
                    outputs = self._model(
                        **inputs, output_hidden_states=True
                    )
                    last_hidden = outputs.hidden_states[-1]
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    embedding = embedding.squeeze(0).cpu()

                domain_embeddings.setdefault(sq["domain"], []).append(embedding)

            self._domain_centroids = {}
            for domain, embeddings in domain_embeddings.items():
                stacked = torch.stack(embeddings)
                centroid = stacked.mean(dim=0)
                self._domain_centroids[domain] = centroid

            logger.info(
                "Computed centroids for %d domains.",
                len(self._domain_centroids),
            )

    @property
    def domain_centroids(self) -> dict:
        return dict(self._domain_centroids)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _cleanup(self) -> None:
        """Free GPU/MPS memory from the current model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded_adapters = []
        self._merged_loaded = False
        self._current_scale = None
        self._domain_centroids = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Previous model cleaned up.")


# Module-level singleton
engine = InferenceEngine()

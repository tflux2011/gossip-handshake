"""Measure routing latency for the paper."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_id, trust_remote_code=True, dtype=torch.float32)

q = "What concentration of neem oil deters the Silver-Back Locust?"
inputs = tok(q, return_tensors="pt")

# Warm up
with torch.no_grad():
    model(**inputs, output_hidden_states=True)

# Router forward pass
times = []
for _ in range(10):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

router_ms = sum(times) / len(times)
print(f"Router forward pass: {router_ms:.1f}ms (mean over 10 runs)")

# Adapter hot-swap
peft_model = PeftModel.from_pretrained(
    model, "./adapters/agronomy_expert_lora/", adapter_name="agronomy")
peft_model.load_adapter(
    "./adapters/veterinary_expert_lora/", adapter_name="veterinary")

swap_times = []
for _ in range(20):
    t0 = time.perf_counter()
    peft_model.set_adapter("agronomy")
    t1 = time.perf_counter()
    swap_times.append((t1 - t0) * 1000)
    t0 = time.perf_counter()
    peft_model.set_adapter("veterinary")
    t1 = time.perf_counter()
    swap_times.append((t1 - t0) * 1000)

swap_ms = sum(swap_times) / len(swap_times)
print(f"Adapter hot-swap: {swap_ms:.2f}ms (mean over 40 swaps)")
print(f"Total routing overhead: {router_ms + swap_ms:.1f}ms")

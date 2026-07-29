"""Spike 0, step 6a: merge the LoRA adapter into the base weights.

Must reload the base model UNQUANTIZED for this - merging a LoRA delta into 4-bit
weights silently loses most of the adapter. fp16 (not bf16: no native bf16 on this
GPU) on CPU, since a 4B-param fp16 model (~8GB) doesn't fit in 6GB VRAM anyway and
this only needs to happen once.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "training/runs/spike0/adapter"
MERGED_DIR = "training/runs/spike0/merged"

print("Loading base model in fp16 on CPU (one-time, slower, needed for a clean merge)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cpu",
)

print("Loading adapter and merging...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged = model.merge_and_unload()

merged.save_pretrained(MERGED_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(MERGED_DIR)

print("Merged model saved to", MERGED_DIR)

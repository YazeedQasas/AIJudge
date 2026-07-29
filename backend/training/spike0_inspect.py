"""Spike 0, step 3: load the student model and see what we actually have.

Don't trust `target_modules="all-linear"` - print the real module names so the LoRA
config in spike0_train.py names them explicitly. Loads in 4-bit (bitsandbytes) because
this model's raw weights don't fit in 6GB VRAM any other way on this GPU.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

config = AutoConfig.from_pretrained(MODEL_ID)
print("architectures:", config.architectures)
print("model_type:", config.model_type)
print("hidden_size:", config.hidden_size, "num_hidden_layers:", config.num_hidden_layers)

# Turing (GTX 1660 Ti, compute capability 7.5) has no native bf16 - fp16 compute dtype.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
)

print()
print("model class:", type(model).__name__)
print()
print("linear module names (for target_modules):")
seen = set()
for name, module in model.named_modules():
    cls = type(module).__name__
    if "Linear" in cls or "4bit" in cls:
        leaf = name.rsplit(".", 1)[-1]
        seen.add(leaf)
print(sorted(seen))

print()
print("VRAM after load (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))

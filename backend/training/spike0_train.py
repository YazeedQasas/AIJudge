"""Spike 0, steps 4-5: attach a rank-4 LoRA and train 10 steps on garbage data.

Quality is irrelevant here - this only proves the plumbing survives: 4-bit base +
LoRA adapter + a training step + a save, on this GPU, for this model. The real
dataset and hyperparameters come later (Stage B/C); this is throwaway.
"""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUT_DIR = "training/runs/spike0/adapter"

# 5 garbage lines - just needs to be well-formed text, not correct or meaningful.
GARBAGE = [
    "The quick brown fox jumps over the lazy dog.",
    "Question: what is 2+2? Answer: purple elephants dance at midnight.",
    "In a hole in the ground there lived a spike zero test.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "To be or not to be, that is a garbage training example.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
    dtype=torch.float16,  # Turing has no native bf16 - without this, PEFT's new
                          # LoRA layers inherit the checkpoint's default bf16 and
                          # crash the fp16 GradScaler on gradient unscale.
)
model = prepare_model_for_kbit_training(model)

# Real module names from spike0_inspect.py - not "all-linear". lm_head excluded,
# LoRA doesn't target it.
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Belt-and-suspenders: whatever dtype PEFT gave the new LoRA A/B matrices (observed
# bfloat16 despite the fp16 base - likely inherited from the checkpoint's declared
# config.torch_dtype), force it to fp16. GradScaler has no bf16 kernel at all - not
# a precision choice, a hard NotImplementedError on this GPU. Only touches params
# that require grad, so the packed 4-bit base weights are untouched.
bf16_params = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype == torch.bfloat16]
print(f"Trainable params found in bfloat16 (forcing to fp16): {len(bf16_params)}")
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float16)

dataset = Dataset.from_dict({"text": GARBAGE})

sft_config = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=10,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    packing=False,
    # No fp16/GradScaler: something in the forward pass produces a bf16 tensor
    # regardless of parameter dtype (root cause not worth chasing for a 10-step
    # spike), and GradScaler has no bf16 kernel at all - not a precision choice,
    # a hard NotImplementedError. 8M trainable params in fp32 is trivial memory.
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print()
print("Saved adapter to", OUT_DIR)
print("Peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))

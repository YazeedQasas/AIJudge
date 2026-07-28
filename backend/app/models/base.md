---
id: base
name: Gemma 4 E4B (base, no fine-tuning)
# The exact identifier LM Studio serves this under. Get it from:
#   curl http://localhost:1234/v1/models
# If it doesn't match byte-for-byte, LM Studio silently serves whatever is loaded
# instead, and an eval comparing two models quietly compares one model to itself.
model: google/gemma-4-e4b
kind: base
order: 0

description: >
  The stock instruction-tuned model as served by LM Studio. This is the control
  arm: every fine-tuned model is measured against it on the same cases, the same
  prompt version, and the same seed.
---

This card exists so that "the base model" is a named thing the eval can vary,
rather than an implicit default buried in config.

Nothing about this model is ours. It is whatever Google shipped, quantized by
whoever built the GGUF, loaded by LM Studio. That is exactly what makes it a
useful baseline — it is the answer to "what do we get for free, before any of
our work?"

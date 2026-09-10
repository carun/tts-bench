# Qwen3-TTS 0.6B.
#
# qwen-tts pins transformers==4.57.3, which is why this cannot share an image
# with NeMo. Isolation is the whole point of the split; do not "fix" the
# resulting downgrade warning by relaxing the pin, because the model code is
# written against that API.
#
# flash-attn is deliberately absent: no aarch64 wheel exists, and building it
# takes longer than the benchmark it would accelerate. engines/qwen.py asks for
# sdpa instead and the report says so.
FROM tts-bench-base:latest

RUN pip install --no-cache-dir qwen-tts

COPY harness /harness
COPY engines /engines

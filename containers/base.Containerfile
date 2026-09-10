# Shared base for every engine image.
#
# The engines have mutually incompatible dependency pins -- qwen-tts wants
# transformers==4.57.3, NeMo wants something much newer -- so each gets its own
# image. What they CAN share is the expensive part: torch for aarch64 with CUDA,
# which is several GB and identical for all of them.
#
# gcc and g++ are NOT optional build tooling here; they are a RUNTIME
# dependency. MagpieTTS JIT-compiles Triton kernels during inference, and on a
# slim image that fails with "RuntimeError: Failed to find C compiler" from
# inside triton, minutes after the model has loaded fine. Removing these to
# slim the image will break exactly one engine, at inference, not at build.
#
# espeak-ng is misaki's fallback G2P for out-of-vocabulary words, which is how
# Kokoro pronounces anything not in its dictionary.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        espeak-ng \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# torch first and alone: it is the one download worth caching as its own layer,
# and letting a later resolver drag in a different build would silently swap the
# CUDA version out from under sm_121.
RUN pip install --no-cache-dir torch torchaudio

RUN pip install --no-cache-dir \
        numpy \
        scipy \
        soundfile \
        librosa \
        whisper_normalizer

ENV HF_HOME=/cache
WORKDIR /work

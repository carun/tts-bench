# Shared container invocation. Sourced, not executed.
#
# An array rather than a string: the run command carries paths, and a string
# built for word-splitting breaks the moment one of them contains a space.
BENCH_WORK="${BENCH_WORK:-$HOME/tts-bench-work}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/tts-bench-hf}"
mkdir -p "$BENCH_WORK"/{out,results,scored} "$HF_CACHE"

RUN=(podman run --rm
     --device nvidia.com/gpu=all
     -e HF_HOME=/cache
     -v "$HF_CACHE:/cache"
     -v "$BENCH_WORK:/work")

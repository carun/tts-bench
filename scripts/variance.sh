#!/bin/bash
# Re-run the sampling engines under different seeds.
#
# Magpie, Qwen and Chatterbox are autoregressive and sample, so one run is a draw rather
# than a measurement. Deterministic engines are skipped: reseeding FastPitch or
# Kokoro produces byte-identical audio and would only pad the table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

for seed in 7 99; do
  echo "### seed $seed"
  "${RUN[@]}" -e "TTS_BENCH_SEED=$seed" tts-bench-nemo:latest python /engines/run_magpie.py
  "${RUN[@]}" -e "TTS_BENCH_SEED=$seed" tts-bench-qwen:latest python /engines/run_qwen.py
  "${RUN[@]}" -e "TTS_BENCH_SEED=$seed" tts-bench-chatterbox:latest python /engines/run_chatterbox.py --variant original
  "${RUN[@]}" -e "TTS_BENCH_SEED=$seed" tts-bench-chatterbox:latest python /engines/run_chatterbox.py --variant turbo
done

for eng in magpie-s7 magpie-s99 qwen3-tts-0.6b-s7 qwen3-tts-0.6b-s99 \
           chatterbox-s7 chatterbox-s99 chatterbox-turbo-s7 chatterbox-turbo-s99; do
  "${RUN[@]}" tts-bench-nemo:latest python /harness/judge.py \
      --results "/work/results/$eng.json" --judge whisper \
      --out     "/work/scored/$eng.whisper.json"
done
echo "### variance complete"

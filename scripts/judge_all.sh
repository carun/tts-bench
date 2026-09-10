#!/bin/bash
# Transcribe every synthesised utterance with both judges and score WER.
#
# Two judges from different families on purpose: Parakeet is NVIDIA's, and so
# are two of the engines under test. Whisper is the disinterested one.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

for res in "$BENCH_WORK"/results/*.json; do
  eng="$(basename "$res" .json)"
  for judge in parakeet whisper; do
    echo "### $eng / $judge"
    "${RUN[@]}" tts-bench-nemo:latest python /harness/judge.py \
        --results "/work/results/$eng.json" \
        --judge   "$judge" \
        --out     "/work/scored/$eng.$judge.json"
  done
done
echo "### judging complete"

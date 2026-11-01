#!/bin/bash
# Build the three engine images. Run from the repo root.
#
# One base image carries torch (several GB, identical for every engine); the
# three engine images differ only in their mutually incompatible pins, which is
# the reason they cannot share one image at all.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
podman build -t tts-bench-base:latest -f containers/base.Containerfile .
for i in kokoro qwen nemo chatterbox; do
  podman build -t "tts-bench-$i:latest" -f "containers/$i.Containerfile" .
done
echo "built: tts-bench-{base,kokoro,qwen,nemo,chatterbox}"

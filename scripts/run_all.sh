#!/bin/bash
# Synthesise the whole corpus with every engine. Each runs in its own image
# because their dependency pins genuinely cannot coexist.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

echo "### fastpitch (default G2P config)"
"${RUN[@]}" tts-bench-nemo:latest   python /engines/run_fastpitch.py --variant default
echo "### fastpitch (ignore_ambiguous_words=False + OOV dictionary)"
"${RUN[@]}" tts-bench-nemo:latest   python /engines/run_fastpitch.py --variant fixed
echo "### kokoro"
"${RUN[@]}" tts-bench-kokoro:latest python /engines/run_kokoro.py
echo "### magpie"
"${RUN[@]}" tts-bench-nemo:latest   python /engines/run_magpie.py
echo "### magpie (+ text normalisation)"
"${RUN[@]}" tts-bench-nemo:latest   python /engines/run_magpie.py --tn
echo "### qwen3-tts"
"${RUN[@]}" tts-bench-qwen:latest   python /engines/run_qwen.py
echo "### chatterbox (original, 0.5B)"
"${RUN[@]}" tts-bench-chatterbox:latest python /engines/run_chatterbox.py --variant original
echo "### chatterbox (turbo, 350M)"
"${RUN[@]}" tts-bench-chatterbox:latest python /engines/run_chatterbox.py --variant turbo
echo "### sweep complete"

# Kokoro-82M. The only engine here whose dependencies are unpinned and simply
# resolve against whatever torch is already present, which is why this is the
# shortest file in the directory.
FROM tts-bench-base:latest

RUN pip install --no-cache-dir kokoro "misaki[en]"

COPY harness /harness
COPY engines /engines

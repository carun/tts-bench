# Chatterbox (Resemble AI), installed --no-deps over the shared torch.
#
# chatterbox-tts 0.1.7 pins torch==2.6.0 and torchaudio==2.6.0 on Python <3.14.
# torch 2.6 has no Blackwell kernels, so honouring that pin on a GB10 means no
# GPU at all. The package itself goes in with --no-deps and its dependencies are
# then installed by hand with torch LOCKED to the base image's build via a
# constraints file. If anything else in the graph insists on torch 2.6, the
# resolver fails here, loudly, instead of quietly swapping the CUDA build out
# from under sm_121.
#
# Every other pin is honoured exactly -- numpy<2, librosa==0.11.0,
# transformers==5.2.0, diffusers==0.29.0, safetensors==0.5.3, conformer==0.3.2
# -- so the only deviation from the released package is torch/torchaudio. That
# includes downgrading numpy, scipy and librosa below what the base ships; that
# is deliberate, not drift. No source file in chatterbox is patched.
#
# Three declared dependencies are left out because nothing on the English path
# imports them: gradio (the demo apps only), and pykakasi / spacy-pkuseg
# (imported lazily, inside the Japanese and Chinese branches of the
# multilingual tokenizer).
#
# setuptools<82 is the one dependency nobody declares. resemble-perth (the
# watermarker every generate() call runs) does `from pkg_resources import ...`
# but lists no requirements at all, and its __init__ swallows the ImportError
# and sets PerthImplicitWatermarker = None. Chatterbox then calls it, and the
# load dies with "TypeError: 'NoneType' object is not callable" -- naming
# neither setuptools nor perth. python:3.12-slim ships no setuptools, and
# setuptools 82+ no longer contains pkg_resources, so "pip install setuptools"
# alone does not fix it either. This is independent of the torch pin: the
# official torch-2.6 install hits it on the same base image.
FROM tts-bench-base:latest

RUN pip freeze | grep -E '^(torch|torchaudio)==' > /etc/torch-lock.txt \
 && cat /etc/torch-lock.txt

RUN pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7

RUN pip install --no-cache-dir -c /etc/torch-lock.txt \
        "numpy>=1.24.0,<2.0.0" \
        "librosa==0.11.0" \
        "transformers==5.2.0" \
        "diffusers==0.29.0" \
        "safetensors==0.5.3" \
        "conformer==0.3.2" \
        "resemble-perth>=1.0.0" \
        s3tokenizer \
        pyloudnorm \
        omegaconf \
        "setuptools<82"

COPY harness /harness
COPY engines /engines

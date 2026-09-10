# FastPitch + HiFiGAN, MagpieTTS, and both ASR judges (Parakeet, Whisper).
#
# The language packages are not optional for an English-only benchmark.
# MagpieTTS instantiates the G2P for all twelve supported languages at load, so
# a missing Chinese or Japanese dependency stops an English run dead -- and it
# reports that as an UnboundLocalError on a variable name from inside hydra,
# never as "install jieba".
#
# pyopenjtalk-plus rather than pyopenjtalk: upstream publishes a source tarball
# only, so plain pyopenjtalk means compiling a C++ extension on aarch64. The
# -plus fork ships a cp312 manylinux aarch64 wheel and provides the same import
# name.
FROM tts-bench-base:latest

RUN pip install --no-cache-dir "nemo_toolkit[asr]" transformers

# nltk carries the CMUdict that FastPitch's EnglishG2p needs. Without it the
# failure is an UnsafeTargetError claiming the target is blocked to prevent
# arbitrary code execution, which is not what is wrong.
RUN pip install --no-cache-dir \
        nltk \
        jieba \
        pypinyin \
        pypinyin_dict \
        pyopenjtalk-plus \
        pykakasi \
        kaldialign

COPY harness /harness
COPY engines /engines

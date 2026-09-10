"""The test corpus.

Seven categories, eight sentences each. The split is the point: a single
aggregate WER hides the fact that engines fail in *different places*, and the
interesting result in TTS is almost never the mean.

Design rules, so the numbers mean something:

  * Every sentence is one utterance a real system might actually be asked to
    say. No tongue-twisters, no minimal pairs, no synthetic stress tests --
    those measure a thing nobody ships.
  * Reference text is written the way a caller would write it, digits and all.
    Normalisation happens at scoring time, identically on both sides, rather
    than by pre-chewing the input into words. An engine that cannot say "1,024"
    should lose points for that; that is the whole job.
  * Length is bounded to roughly one breath. Long-form synthesis is a separate
    axis (chunking, prosody drift) and mixing it in would confound the read.

`heteronyms` deserves its own note. Every sentence there contains a word whose
pronunciation depends on meaning -- "read", "lead", "live", "bass". A
dictionary lookup alone cannot resolve them; the frontend needs part-of-speech
or context. This is the category that separates a real G2P from a lookup table,
and it is where the FastPitch `ignore_ambiguous_words` default does its damage:
words with more than one pronunciation get dropped to raw letters.
"""

CORPUS = {
    "prose": [
        "The meeting was moved to the smaller room on the second floor.",
        "I never thought the answer would turn out to be that simple.",
        "She left a note on the counter saying she would be back by six.",
        "Most of the work happens before anyone writes a single line of code.",
        "There is a difference between being busy and being useful.",
        "He asked whether the results had been checked by anyone else.",
        "The weather turned just as we were about to leave the house.",
        "Nobody had told her the deadline had already been extended twice.",
    ],
    "numbers": [
        "The invoice came to $1,247.50 including tax.",
        "Our meeting is at 3:45 PM on March 22nd.",
        "Revenue grew 18.5% compared to the same quarter in 2024.",
        "The file was 1,024 megabytes, which took 47 minutes to upload.",
        "Call me back on 555-0143 after 9 AM.",
        "Only 3 of the 128 test cases failed on the first run.",
        "The building was finished in 1897 and renovated in 2011.",
        "Set the timeout to 30 seconds and the retry count to 5.",
    ],
    "acronyms": [
        "The GPU ran out of VRAM during the second epoch.",
        "Send the PDF to the IT department before EOD.",
        "NASA confirmed the launch window opens in April.",
        "The API returns HTTP 429 when you exceed the rate limit.",
        "She works for the FBI, not the CIA.",
        "Our SQL queries hit the CDN cache about 80% of the time.",
        "The NVIDIA driver needs a reboot before CUDA will initialise.",
        "Attach the CSV and the JSON export to the same email.",
    ],
    "jargon": [
        "The inference sidecar queues contributions until the GPU responds.",
        "Speaker diarization splits the recording into turns before transcription.",
        "We rolled back the Kubernetes deployment after the readiness probe failed.",
        "The embeddings are stored in a vector index rather than a relational table.",
        "Quantisation shrank the checkpoint without measurably hurting accuracy.",
        "Backpropagation through the encoder was disabled during fine-tuning.",
        "The tokenizer emitted graphemes instead of phonemes for ambiguous words.",
        "Our autoscaler thrashed because the readiness threshold was too aggressive.",
    ],
    "names": [
        "Siobhan and Xiaoling met Przemysław at the conference in Reykjavik.",
        "Dr. Nguyen reviewed the manuscript before Professor Okonkwo signed off.",
        "We drove from Worcester to Gloucester and then on to Leicester.",
        "Ask Aoife whether Sanjay has heard back from Mercedes.",
        "The paper was co-authored by Bhattacharya, Lindqvist and Oyelaran.",
        "Yosemite in November is quieter than Yellowstone in July.",
        "Giannis passed to Antetokounmpo's teammate in the final seconds.",
        "Her flight connects through Ouagadougou and then Nairobi.",
    ],
    "complex": [
        "Although the tests passed locally, the build failed in CI, which nobody noticed until Monday.",
        "The report, which had been delayed twice already, finally arrived without the appendix.",
        "If the connection drops mid-upload, the client retries; otherwise, it reports success immediately.",
        "What surprised us was not that it failed, but how long it took anyone to notice.",
        "He said, quite reasonably, that the deadline had never been agreed in writing.",
        "The system works — assuming, of course, that the cache has been warmed first.",
        "Before deploying, verify the image was rebuilt; a stale image looks identical from outside.",
        "She wanted to know why, given everything that had happened, we were still shipping on Friday.",
    ],
    "heteronyms": [
        "I read the report yesterday, and I will read the summary tonight.",
        "The lead engineer removed the lead pipe from the basement.",
        "They live near the studio where the live broadcast is recorded.",
        "He plays bass guitar, though he would rather be fishing for bass.",
        "Please close the door; the exit is close to the stairwell.",
        "The dove dove straight into the hedge without slowing down.",
        "We must record the meeting and file the record with the board.",
        "The wind was strong enough to wind the flag around the pole.",
    ],
}


def flatten():
    """[(id, category, text)] in a stable order."""
    out = []
    for cat in sorted(CORPUS):
        for i, text in enumerate(CORPUS[cat]):
            out.append({"id": f"{cat}_{i:02d}", "category": cat, "text": text})
    return out


if __name__ == "__main__":
    import json
    import sys

    items = flatten()
    json.dump(items, sys.stdout, indent=2, ensure_ascii=False)
    print(f"\n# {len(items)} sentences across {len(CORPUS)} categories", file=sys.stderr)

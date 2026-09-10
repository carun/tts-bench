## Speed

| Engine | Params | Load (s) | RTF median | RTF p90 | × real-time | Peak GPU | SR |
|---|---|---|---|---|---|---|---|
| fastpitch-default | ~55M (FastPitch) + ~85M (HiFiGAN) | 18.9 | 0.0085 | 0.0092 | 117.6× | 729 MiB | 22050 |
| fastpitch-fixed | ~55M (FastPitch) + ~85M (HiFiGAN) | 19.5 | 0.0085 | 0.0091 | 117.6× | 729 MiB | 22050 |
| kokoro | 82M | 4.4 | 0.0126 | 0.0135 | 79.4× | 707 MiB | 24000 |
| magpie | 357M + NanoCodec | 31.3 | 0.4119 | 0.4168 | 2.4× | 2345 MiB | 22050 |
| magpie-tn | 357M + NanoCodec | 30.4 | 0.4166 | 0.4207 | 2.4× | 2352 MiB | 22050 |
| qwen3-tts-0.6b | 0.6B | 19.8 | 0.5879 | 0.5910 | 1.7× | 2301 MiB | — |

## Intelligibility — Parakeet judge

| Engine | Overall WER | prose | numbers | acronyms | jargon | names | complex | heteronyms |
|---|---|---|---|---|---|---|---|---|
| kokoro | **3.1%** | 0.0% | 8.6% | 0.0% | 2.5% | 13.0% | 0.0% | 0.0% |
| qwen3-tts-0.6b | **6.7%** | 0.0% | 21.0% | 2.5% | 8.9% | 16.9% | 0.0% | 2.2% |
| magpie-tn | **7.0%** | 0.0% | 27.2% | 3.8% | 2.5% | 18.2% | 1.8% | 0.0% |
| magpie | **7.1%** | 0.0% | 27.2% | 7.6% | 3.8% | 14.3% | 1.8% | 0.0% |
| fastpitch-default | **8.8%** | 1.1% | 21.0% | 19.0% | 2.5% | 16.9% | 3.5% | 2.2% |
| fastpitch-fixed | **10.2%** | 2.1% | 33.3% | 19.0% | 2.5% | 19.5% | 1.8% | 0.0% |

## Intelligibility — Whisper judge

| Engine | Overall WER | prose | numbers | acronyms | jargon | names | complex | heteronyms |
|---|---|---|---|---|---|---|---|---|
| kokoro | **2.3%** | 0.0% | 4.9% | 0.0% | 6.3% | 5.2% | 0.9% | 0.0% |
| qwen3-tts-0.6b | **4.1%** | 0.0% | 14.8% | 1.3% | 8.9% | 5.2% | 0.9% | 0.0% |
| magpie-tn | **5.9%** | 0.0% | 23.5% | 3.8% | 3.8% | 10.4% | 2.6% | 0.0% |
| magpie | **7.0%** | 0.0% | 28.4% | 5.1% | 5.1% | 11.7% | 2.6% | 0.0% |
| fastpitch-default | **7.8%** | 0.0% | 24.7% | 16.5% | 6.3% | 7.8% | 2.6% | 1.1% |
| fastpitch-fixed | **9.4%** | 0.0% | 33.3% | 17.7% | 3.8% | 10.4% | 4.4% | 1.1% |

## Judge agreement

| Engine | Parakeet WER | Whisper WER | Δ |
|---|---|---|---|
| fastpitch-default | 8.8% | 7.8% | 1.0% |
| fastpitch-fixed | 10.2% | 9.4% | 0.8% |
| kokoro | 3.1% | 2.3% | 0.8% |
| magpie | 7.1% | 7.0% | 0.2% |
| magpie-tn | 7.0% | 5.9% | 1.1% |
| qwen3-tts-0.6b | 6.7% | 4.1% | 2.6% |

## Worst individual utterances (Whisper judge)

| Engine | Cat | WER | Asked | Heard |
|---|---|---|---|---|
| qwen3-tts-0.6b | numbers | 55.6% | Call me back on 555-0143 after 9 AM. | Call me back on 505-AV-100-0143 after 9 a.m. |
| fastpitch-default | numbers | 50.0% | Our meeting is at 3:45 PM on March 22nd. | Our meeting is at p.m. on Marchend. |
| fastpitch-fixed | numbers | 50.0% | Our meeting is at 3:45 PM on March 22nd. | Our meeting is at p.m. on Marchend. |
| fastpitch-fixed | numbers | 50.0% | The building was finished in 1897 and renovated in 2011. | The building was finished and unrenovated and |
| fastpitch-fixed | acronyms | 45.5% | Attach the CSV and the JSON export to the same email. | Attach the XenXport to the same email. |
| fastpitch-default | numbers | 44.4% | Call me back on 555-0143 after 9 AM. | Call me back on after M. |
| fastpitch-fixed | numbers | 44.4% | Call me back on 555-0143 after 9 AM. | Call me back on after M. |
| magpie | numbers | 44.4% | Call me back on 555-0143 after 9 AM. | Call me back on eeeee after a.m. |
| magpie-tn | numbers | 44.4% | Call me back on 555-0143 after 9 AM. | Call me back on after a.m. |
| fastpitch-default | acronyms | 41.7% | Our SQL queries hit the CDN cache about 80% of the time. | Our school queries hit the kinkachib out of the time. |
| fastpitch-fixed | numbers | 41.7% | Set the timeout to 30 seconds and the retry count to 5. | Set the timeout 2 seconds under the retry count 2. |
| fastpitch-fixed | names | 40.0% | Siobhan and Xiaoling met Przemysław at the conference in Reykjavik. | Chaubon und Zeiling met Przemysl at the conference in Reykjavik. |

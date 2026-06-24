# Real-Time Gaming Voice Toxicity Detector

Online gaming has a toxicity problem. This project tries to detect it automatically — not just from what someone says, but how they say it.

Most text-based toxicity detectors miss the audio side entirely. Someone can type calmly while screaming into their mic. This pipeline handles both.

---

## How it works

The pipeline runs five things in parallel and fuses them into a single toxicity score:

**1. Speech-to-text** — OpenAI Whisper transcribes the audio input.

**2. Text toxicity scoring** — A fine-tuned transformer (`martin-ha/toxic-comment-classification`) classifies the transcript for toxic content.

**3. Emotion detection** — Wav2Vec2 (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) analyzes the raw audio and returns an emotion label (angry, happy, neutral, sad, other).

**4. Voice energy analysis** — Librosa computes the RMS energy of the audio. Loud = potentially aggressive. This feeds into a shouting score.

**5. Context feature extraction** — A custom rule-based layer on top of the transcript that catches profanity density, personal attack patterns ("you are", "you're", "ur"), all-caps aggression, and word repetition.

These five signals go into an **adaptive fusion engine** that adjusts weights dynamically. If the voice is loud and the emotion model is confident about anger, rage weight goes up. If emotion confidence is weak, text toxicity gets more weight. It's not a static formula.

There's also a **session memory layer** — a rolling deque of the last 20 scores — that tracks escalation over time. Someone getting progressively louder and more aggressive across multiple clips gets flagged differently than a single outburst.

---

## Output

For each audio input, the interface returns:

- Transcribed text
- Toxicity label (TOXIC / NOT TOXIC / UNKNOWN)
- Text toxicity score
- Voice energy (shouting) score
- Detected emotion
- Rage detection score
- Escalation score
- Final fused toxicity score

---

## The Gradio interface

Upload or record an audio clip and get the full breakdown in real time. Built with Gradio, runs in Google Colab with a public share link.

---

## Models used

- `openai/whisper-base` — speech transcription
- `martin-ha/toxic-comment-classification` — text toxicity
- `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` — speech emotion recognition

All pulled from Hugging Face. No custom training needed.

---

## Limitations

The emotion model has some known weight mismatches on load — it still works but the confidence scores can be unreliable on certain audio conditions. When confidence drops below 0.4, the pipeline automatically reduces the emotion signal weight and leans harder on text.

The toxic word list in the context feature extractor is small and gaming-specific. It catches common gaming toxicity ("noob", "dogwater", "ez") but won't generalize well to other domains without expanding the list.

---

## Stack

- Python, PyTorch, Transformers (Hugging Face)
- OpenAI Whisper, SpeechBrain, Librosa
- Gradio
- Google Colab (GPU runtime recommended)

---

## How to run

Open in Google Colab, switch to a GPU runtime (Runtime → Change runtime type → T4 GPU), and run cells top to bottom. Models download automatically from Hugging Face on first run. The Gradio interface launches with a public link at the end.

First run takes a few minutes — Whisper and Wav2Vec2 are not small.

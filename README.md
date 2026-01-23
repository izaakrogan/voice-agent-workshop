# Exercise 3 (Bonus): Audio Representations for LLMs

**Understanding why audio needs better representations**

When we want to build LLMs that truely understand and generate speech, we hit a fundamental problem: audio is *hard* to model compared to text. This workshop explores why, through hands-on PyTorch excercises.

## The Problem

Text LLMs work beautifully. Take a lot of text, a big Transformer, train it to predict the next token, and you get remarkably coherent outputs.

But try the same approach with audio and you get... nonsense.

Why? The answer lies in the *density* of information. A single second of audio contains tens of thousands of samples, but corresponds to only a few words. This creates two problems:

1. **Scale**: Models struggle to maintain coherence over such long sequences
2. **Speed**: Generating audio sample-by-sample is painfully slow

This workshop will help you build intuition for these challenges through code.

## Setup

```bash
pip install torch torchaudio matplotlib numpy librosa
```

## Excercise 1: The Scale of Audio

Let's start by understanding just how much data we're dealing with.

```python
import torch
import torchaudio
import matplotlib.pyplot as plt

sample_rate = 16000
duration_seconds = 10

t = torch.linspace(0, duration_seconds, sample_rate * duration_seconds)
audio = torch.sin(2 * torch.pi * 200 * t) * torch.sin(2 * torch.pi * 3 * t)

print(f"Audio duration: {duration_seconds} seconds")
print(f"Sample rate: {sample_rate} Hz")
print(f"Total samples: {len(audio):,}")
print(f"Samples per second: {sample_rate:,}")
```

**Question**: If this 10-second clip contains roughly 10 words of speech, how many audio samples represent each word on average?

<details>
<summary>Answer</summary>

With 160,000 samples for ~10 words, that's roughly **16,000 samples per word**. Compare this to text, where a word might be 1–3 tokens. The ratio is staggering.

</details>

## Excercise 2: Text vs Audio Information Density

Let's make the comparision concrete.

```python
text = "The quick brown fox jumps over the lazy dog"
text_chars = len(text)
text_tokens_approx = len(text.split())

words = len(text.split())
audio_duration = words * 0.5
audio_samples = int(audio_duration * sample_rate)

print("Text representation:")
print(f"  Characters: {text_chars}")
print(f"  Words (≈ tokens): {text_tokens_approx}")

print("\nAudio representation (16kHz):")
print(f"  Duration: {audio_duration} seconds")
print(f"  Samples: {audio_samples:,}")

print(f"\nRatio: {audio_samples / text_tokens_approx:.0f}x more values in audio than text tokens")
```

## Excercise 3: Visualising Audio Representations

Raw audio samples are just numbers. Let's see what they look like and explore richer representatons.

```python
import numpy as np

sample_rate = 16000
duration = 1.0
t = torch.linspace(0, duration, int(sample_rate * duration))

audio = (
	0.5 * torch.sin(2 * torch.pi * 150 * t) +
	0.3 * torch.sin(2 * torch.pi * 300 * t) +
	0.2 * torch.sin(2 * torch.pi * 450 * t)
)

envelope = 0.5 + 0.5 * torch.sin(2 * torch.pi * 2 * t)
audio = audio * envelope

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t[:1600], audio[:1600])
axes[0].set_title("Raw Waveform (first 100ms)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

axes[1].plot(t[:160], audio[:160], 'o-', markersize=2)
axes[1].set_title("Individual Samples (first 10ms) – This is what the model sees")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")

spectrogram = torchaudio.transforms.Spectrogram(
	n_fft=512,
	hop_length=128,
	power=2
)
spec = spectrogram(audio.unsqueeze(0))
axes[2].imshow(
	torch.log(spec[0] + 1e-9), 
	aspect='auto', 
	origin='lower',
	extent=[0, duration, 0, sample_rate/2]
)
axes[2].set_title("Spectrogram – frequency content over time")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig("audio_representations.png", dpi=150)
plt.show()
```

**Key insight**: The spectrogram reveals structure (frequency patterns) that isn't obvious from raw samples. This hints at why we need *learnt* representations that capture meaningful audio structure.

## Excercise 4: The Context Window Problem

LLMs have limited context windows. Let's see how this constrains audio modelling.

```python
context_sizes = {
	"GPT-2 small": 1024,
	"GPT-2 medium": 2048,
	"Modern LLMs": 8192,
}

sample_rate = 16000

print("How much audio fits in different context windows?\n")
print(f"{'Model':<20} {'Context':<10} {'Audio Duration':<15} {'Approx Words':<15}")
print("-" * 60)

for model, ctx in context_sizes.items():
	duration_ms = (ctx / sample_rate) * 1000
	approx_words = duration_ms / 500
	print(f"{model:<20} {ctx:<10} {duration_ms:.0f}ms{'':<10} ~{approx_words:.1f} words")

print("\n⚠️  With sample-by-sample modelling, even 8k context only covers ~0.5 seconds!")
print("   That's not even enough for a complete sentence.")
```

## Excercise 5: Why Sample-by-Sample Fails

Let's simulate what happens when a model can only "see" a tiny window of audio.

```python
def simulate_context_window(audio, context_size, sample_rate):
	duration_visible = context_size / sample_rate
	samples_visible = min(context_size, len(audio))
	return audio[:samples_visible], duration_visible

sample_rate = 16000
full_audio = torch.randn(5 * sample_rate)

context_size = 2048
visible_audio, visible_duration = simulate_context_window(
	full_audio, context_size, sample_rate
)

print(f"Full audio: {len(full_audio)/sample_rate:.1f} seconds ({len(full_audio):,} samples)")
print(f"Model sees: {visible_duration*1000:.0f}ms ({len(visible_audio):,} samples)")
print(f"Model is blind to: {100 * (1 - len(visible_audio)/len(full_audio)):.1f}% of the audio")
```

**The problem**: With only 128ms of context, the model can't learn that sentences have subjects and predicates, that speakers maintain consistant voices, or any other long-range patterns.

## Excercise 6: The Compression Imperative

To model audio effectively, we need compression. Let's explore the target compression ratios.

```python
def calculate_compression_needs(
	target_context_seconds: float,
	model_context_size: int,
	sample_rate: int = 16000
) -> float:
	samples_needed = target_context_seconds * sample_rate
	compression_ratio = samples_needed / model_context_size
	return compression_ratio

print("Compression ratios needed to fit X seconds into a 2048-token context:\n")

for target_seconds in [1, 5, 10, 30, 60]:
	ratio = calculate_compression_needs(
		target_context_seconds=target_seconds,
		model_context_size=2048
	)
	print(f"  {target_seconds:>2}s of audio → {ratio:>6.1f}x compression needed")

print("\n💡 Neural audio codecs like Mimi achieve ~128x compression!")
print("   This means 2048 tokens can represent ~16 seconds of audio.")
```

## Summary

| Representation | Samples/second | 10s audio size | Context coverage (2048 tokens) |
|----------------|----------------|----------------|-------------------------------|
| Raw audio (16kHz) | 16,000 | 160,000 samples | 128ms |
| Neural codec (125 fps) | 125 | 1,250 tokens | 16s |
| Neural codec (12.5 fps) | 12.5 | 125 tokens | 164s |

The path forward is clear: **we need learnt compression** that preserves the important information whilst dramaticaly reducing the sequence length.

This is exactly what neural audio codecs like SoundStream, EnCodec, and Mimi provide. They learn to encode audio into discrete tokens that can be modelled by standard LLM architectures.

## Next Steps

Now that you understand *why* we need better audio representations, the natural next steps are:

1. **Autoencoders**: Learn to compress and reconstruct audio
2. **Vector Quantisation**: Make the compressed representation discrete (LLM-friendly)
3. **Residual Vector Quantisation**: Stack quantisers for better fidelity
4. **Training audio LLMs**: Use the compressed tokens to train generative models

## Refferences

- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) – Karpathy's 2015 post
- [WaveNet](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/) – DeepMind's sample-by-sample audio model
- [SoundStream](https://arxiv.org/abs/2107.03312) – First neural audio codec with RVQ
- [Mimi](https://arxiv.org/abs/2410.00037) – Kyutai's codec used in Moshi

---

*Workshop materials adapted from Kyutai's neural audio codecs blog post. Go and read it, it's very nice*
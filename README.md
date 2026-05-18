# swap-moe

Lightweight MoE (Mixture of Experts) system with dynamic model swapping.
Built to run fully local — no cloud, no internet required.

## Concept

A small router classifies the user's voice input (domain + intent),
saves a checkpoint, then loads only the relevant expert model.
After the expert answers, it unloads — keeping RAM usage minimal.

```
Voice → Whisper Tiny → Router → [checkpoint]
                                     ↓
                          Expert loads → answers → unloads
                                     ↓
                              Answer to user
```

## Architecture
- **Router**: Qwen2.5-0.5B fine-tuned for domain + intent classification
- **Experts**: SmolLM2-360M fine-tuned per domain
- **Swap Manager**: Handles model loading/unloading + context persistence
- **Target**: Runs fully local on Android (<500 MB total)

## Stack
- Apple MLX (training on Apple Silicon)
- llama.cpp (Android inference)
- Python 3.11+

## Status
Work in progress — personal open source project

## License
MIT — see LICENSE

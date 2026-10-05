# Lichess GPT

A small GPT-style Transformer trained on Lichess chess games, built as a controlled environment for studying **mechanistic interpretability**.

The model learns to predict the next chess move from the preceding move history using UCI move-level tokenization.

## Current Model

- **Architecture:** Decoder-only Transformer
- **Parameters:** ~12M
- **Layers:** 6
- **Embedding dimension:** 384
- **Attention heads:** 16
- **Context length:** 256 moves
- **Vocabulary:** ~1.8K UCI moves

Training examples are generated **on demand** from game-level data rather than materializing the entire dataset in memory. Evaluation is performed exhaustively over the train and development splits.

## Goal

The immediate goal is to establish a clean and reproducible Transformer baseline that can later be used to investigate how the model internally represents chess state and performs its computations.

Future work will focus on representation analysis, causal interventions, and mechanistic interpretability experiments.

> **Status:** Early research / work in progress.

*This project is independent and not affiliated with Lichess.*
# Lichess-GPT

A small GPT trained from scratch on Lichess games, used as a controlled testbed for studying the relationship between next-token prediction and autonomous behaviour.

## Research Question

> **Can autoregressive next-move prediction produce robust autonomous legal chess generation?**

The model is trained on legal chess trajectories using autoregressive next-token prediction, then evaluated in closed loop: its own predictions become the context for subsequent predictions.

This separates two questions:

- **Prediction:** how well does the model predict the next move?
- **Autonomy:** how long can it generate legal moves under its own predictions?

## What I Built

I implemented a decoder-only Transformer from scratch, progressing from an MLP baseline through self-attention and multi-head attention to the final Transformer.

The experiments then tested five forms of scaling:

- training duration
- model width
- context length
- model depth
- training-data scale

The final model uses:

- 50K Lichess games
- 12 Transformer blocks
- `d_model = 224`
- 4 attention heads
- context length 128
- ~8.14M parameters
- UCI move tokenization

Autonomous legality is evaluated independently with `python-chess`.

## Result

Scaling improved next-token prediction, but did not produce robust autonomous legality.

The final 50K-game model:

- **Median first illegal move:** 4
- **Survival to 10 moves:** 0%
- **Survival to 20 moves:** 0%
- **Survival to 50 moves:** 0%

Across the tested scaling interventions, the gap between improved prediction and autonomous legality persisted.

This does **not** establish that scaling cannot produce autonomous chess, or that next-token prediction is fundamentally incapable of doing so. It establishes the result only within the tested model, training, and data regime.

## Current Research

Scaling is now frozen.

The next question is:

> **What information and computation has the trained model learned that can explain the gap between improved next-move prediction and failure to sustain autonomous legal chess generation?**

The full research report documents the experimental design, scaling results, and subsequent investigation.

**Full report:** [Research report]
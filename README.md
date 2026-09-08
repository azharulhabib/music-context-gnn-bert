# GNN-Based BERT for Understanding Context from Music

A hybrid Graph Neural Network and BERT system for multi-label music tag understanding, built for CSE425/EEE474/CSE715 (Neural Networks).

**Author:** Azharul Habib | Student ID: 22241009 | BRAC University

## Overview

This project combines two complementary views of a music track:

- **Graph Neural Network (GNN)** — models audio as a graph of temporally and harmonically related segments, built from mel-spectrogram and chroma features.
- **BERT** — encodes textual descriptions (captions or tags) into semantic embeddings.

These two representations are fused via cross-attention to predict genre-related tags, with the hypothesis that structural and semantic information are complementary.

Three of the four assignment tasks were implemented:

- **Task 1:** BERT baseline for caption-to-tag classification
- **Task 2:** GNN baseline (GraphSAGE and GAT variants) for structure-to-tag classification
- **Task 3:** GNN-BERT fusion via cross-attention

Task 4 (contrastive retrieval) was scoped out; see the Limitations section.

## Results Summary

| Model | Macro-F1 | Micro-F1 |
|---|---|---|
| BERT (Task 1, MusicCaps captions) | 0.67 | 0.74 |
| GraphSAGE (Task 2, mel features) | 0.37 | 0.59 |
| GAT (Task 2, mel features) | 0.35 | 0.59 |
| Fusion (Task 3, GraphSAGE + BERT) | 0.64 | 0.73 |

Full details, methodology, and discussion are in the project report.

## Project Structure

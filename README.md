# STDRF: Sketch-Text Driven Rectified Flow for 4D Face Generation

This repository provides the research implementation of **STDRF**, a sketch-text-driven framework for controllable 4D face generation. STDRF generates temporally coherent 4D facial mesh sequences from two complementary conditions: a structural sketch for identity and geometry, and a text prompt for expression and motion semantics.

The project contains the main method components and data-construction utilities used in our manuscript.

## Method Overview

STDRF formulates sketch-text conditioned 4D face generation as a conditional rectified-flow learning problem. Instead of relying on a multi-stage image/video reconstruction pipeline, STDRF directly models the probability transport from Gaussian noise to the 4D facial motion manifold.

The framework consists of three main parts:

1. **Robust sketch encoding.** A sketch encoder with GCTD preprocessing extracts identity-aware structural priors from sparse facial sketches.
2. **Dual-path condition injection.** Sketch features and text semantics are injected through decoupled cross-attention branches, reducing interference between geometric and semantic conditions.
3. **Topology-aware 4D generation.** An ADNB-based geometric denoising backbone with spectral diffusion predicts the rectified-flow velocity field for temporally coherent mesh generation.

The default representation follows a FLAME-style mesh topology with 5023 vertices per frame and a 160-frame output sequence.

## Release Plan

Training scripts, inference scripts, and model resources are being organized and will be released in a future update.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n stdf python=3.10
conda activate stdf
pip install -r requirements.txt
```

Install the PyTorch version that matches your CUDA environment from the official PyTorch website.

## Data Preparation

The code assumes registered 4D facial mesh sequences with consistent topology. The triplet-construction scripts can be used to generate paired sketch, text, and mesh-sequence records from a prepared 4D face dataset.

Typical preprocessing steps include:

1. inspecting abnormal or invalid sequences;
2. resampling mesh sequences to a fixed temporal length;
3. generating sketch maps from mesh renderings;
4. creating hierarchical text annotations;
5. splitting the data by subject identity.

Dataset paths and preprocessing options can be configured in `config.py`.

## Contact

For questions about the method implementation, please open an issue in this repository.

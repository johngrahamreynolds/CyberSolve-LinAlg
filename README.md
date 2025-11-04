# CyberSolve-LinAlg

This repository contains Azure Databricks notebooks for preprocessing, benchmarking, training, and evaluating iterations of the *CyberSolve-LinAlg-1.** family of seq2seq mathematical reasoning models.

## Motivation for the Research

This research project was motivated by a [talk](https://as.vanderbilt.edu/physics-astronomy/colloquium-john-jumper/) given by John Jumper, 2024 Nobel laureate in Chemistry (Google DeepMind), at the Vanderbilt University Department of Physics and Astronomy on August 31st, 2023.

Jumper's talk, entitled "Highly Accurate Protein Structure Prediction and Its Applications", centered around AlphaFold's neural network modeling of protein structure and protein-protein interactions. At the time, I was working as a Data and ML Engineer while remaining active in pure mathematical physics research. The talk planted the seed for exploring theoretical applications of machine learning to advance scientific discovery.

Approximately one year later, while studying string theory and its mathematical breadth, I became interested in whether neural networks, like AlphaFold in the biological sciences, could be applied to the logical domain of mathematical reasoning. Specifically, could an intelligent model learn to understand mathematics? Furthermore, could a neural network or other AI architecture eventually understand and make breakthroughs in challenging areas of mathematics such as string theory and theoretical physics? This research project serves as an initial step toward answering these questions.

I am grateful to John Jumper for his role in motivating this intellectual pursuit.

## Research and Engineering Overview

My research process involved investigating multiple mathematics datasets, including the [Google DeepMind mathematics dataset](https://huggingface.co/datasets/deepmind/math_dataset) and the [Nvidia OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), along with compatible language models such as [FLAN-T5-large](https://huggingface.co/google/flan-t5-large) and [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it).

My goal was to identify large-scale dataset(s) and a pretrained model architecture that would be jointly compatible for fine-tuning to rigorous mathematical analysis. After extensive evaluation, I selected the combination of the DeepMind mathematics dataset with the pretrained Google FLAN-T5-large model as the foundation for experimentation. Guidance was provided by the [DeepMind publication](https://arxiv.org/abs/1904.01557) that introduced the DeepMind mathematics dataset.

## Model Cards

Model cards for CyberSolve-LinAlg are published on [my Hugging Face profile](https://huggingface.co/MarioBarbeque). Available models:
- **Latest version**: [CyberSolve-LinAlg-1.2](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.2)
- **Initial version**: [CyberSolve-LinAlg-1.1](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.1)

## GPU-Optimized Inference

I have developed a dedicated application for NVIDIA GPU-optimized inference with CyberSolve-LinAlg v1.2, available as a Hugging Face Space.

The Space utilizes Python `streamlit`, the NVIDIA `apex` library, and the Hugging Face `transformers` and `tokenizers` ecosystem to provide fast inference through a simple UI. The fine-tuned CyberSolve models are several gigabytes in size and fit comfortably on a single GPU. The inference application runs on an NVIDIA T4.

**GPU-optimized inference**: [CyberSolve-LinAlg-1.2 inference](https://huggingface.co/spaces/MarioBarbeque/CyberSolveLinAlg1.2)

If the Space is inactive, restart it; the compute instance will be ready in less than 5 minutes.

## Notebooks

This section provides links to the notebooks used throughout the research process.

### Preprocessing - [Preprocessing Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/preprocessing_and_inspection.ipynb)

This notebook contains initial analysis steps along with preprocessing and dataset cleaning procedures with considerations for training and evaluation.

In addition to evaluating exact correctness, I assessed partial correctness of model predictions. For example, determining whether the model accurately predicted `21` when the answer was `-21`. This analysis is challenging; I dedicated effort in this notebook to develop a scalable method for evaluating partial correctness across all predictions.

### Benchmarking - [Benchmarking Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/benchmarking.ipynb)

This notebook contains the benchmarking process for the base FLAN-T5-large seq2seq model's mathematical reasoning capability on the DeepMind mathematics evaluation dataset.

As noted above, I also dedicated effort to evaluating the FLAN-T5 model's partial correctness in solving linear equations. This analysis was intended to provide additional insight into the model's underlying mathematical reasoning ability beyond exact match scores. Given the complexity of this task, I include notes on the challenges of defining this process systematically, and proceeded primarily with exactness metrics. I plan to revisit partial correctness analysis in future work.

### Downsampled Training and Evaluation - [Downsampled Training Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/downsampled_training.ipynb)

Before committing to the full, expensive process of fine-tuning on all 2M records in the DeepMind mathematics dataset's 1D linear algebra split, I conducted a downsampled training run to experimentally validate that the FLAN-T5 model would generalize and domain-adapt well to mathematical reasoning tasks.

I encountered several challenges during this initial training process while determining the optimal method for distributed training on Azure cloud-provided NVIDIA GPUs in the Databricks notebook environment. As a result, some warnings were generated and errors occurred after training and evaluation due to distributed command cells failing to terminate properly. Despite this, training and evaluation completed successfully and results were saved to storage. Intermediate outputs for evaluation scores per epoch remain available.

### Full Training and Evaluation - [Full Training and Eval Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/full_training_and_eval.ipynb)

I trained the model on the complete DeepMind mathematics dataset's 1D linear algebra split and subsequently evaluated its mathematical reasoning ability.

As in benchmarking and downsampled training, I made extensive use of the NVIDIA `apex` package for optimizing training and inference across a distributed system of NVIDIA A100 GPUs. Comprehensive hyperparameter details can be found in the [CyberSolve-LinAlg-1.2 model card](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.2).

I began with 3 epochs of training across 2M total records. Results were strong (86.6% exact match score), and given the consistent upward trend during training, I believed additional improvement was possible. As such, I conducted an additional 2 epochs of training with a modified learning rate.

The final CyberSolve-LinAlg-1.2 model checkpoint achieves an exact match score of 90.75% on solving linear equations from the DeepMind mathematics evaluation dataset.

As a final step, I constructed a partial correctness dataset containing predicted tokens, label tokens, decoded predictions, and decoded labels for future analysis of the model's partial correctness in mathematical reasoning.

## Conclusion and Future Work

This project established a foundation for advancing mathematical reasoning capabilities in artificial neural models.

Further work is needed to better understand the extent to which neural models can reason both fully and partially in various mathematical contexts. This work is currently constrained to 1-dimensional linear equations; I believe it would be valuable to expand this base knowledge, as demonstrated in the DeepMind mathematics paper, to evaluate how well models learn different branches of mathematics with varying degrees of rigor and complexity.

I hope to return to this project in the near future to expand CyberSolve's mathematical knowledge base and conduct a detailed retrospective analysis of the model's partial reasoning capability with respect to acquired knowledge.

A significant challenge during this research was the cost of long-running cloud GPU compute, which served as a barrier to further experimentation. I am grateful to Vanderbilt University for providing the opportunity to conduct this research.

My long-term goal remains training artificial neural models to achieve advanced, expert-level understanding of modern mathematics.


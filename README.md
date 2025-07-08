# CyberSolve-LinAlg
This repo contains the Azure Databricks notebooks used for preprocessing, benchmarking, training, and evaluating various iterations of the *CyberSolve-LinAlg-1.** family of seq2seq mathematical reasoning models. Other relevant information is added here as well.

### Motivation for the Research

This research project was [motivated by a talk](https://as.vanderbilt.edu/physics-astronomy/colloquium-john-jumper/) given by John Jumper, 2024 Nobel laureate in Chemistry, Google DeepMind, at the Vanderbilt University Department of Physics and Astronomy on August 31st, 2023. 

JJ's talk, entitled "Highly Accurate Protein Structure Prediction and Its Applications", centered around the AlphaFold neural network's modeling of both protein structure and protein-protein interactions. At the time, I was already working as a Data and ML Engineer, but was still incredibly active in pure mathematical physics research. I truly hadn't yet taken an interest in theoretical applications of ML to make/aid in breakthrough scientific discoveries. JJ's talk planted the seed that began to radically redefine my perception of ML.

After a year or so, with JJ's talk still in my mind, and while studying string theory and its robust mathematical breadth, I became interested in wondering how well NNs, like AlphaFold in the biological and biochemical discipline, could be applied to the more logical domain of mathematical reasoning. In other words, could one teach an intelligent model to understand mathematics? Furthermore, could a NN or some other artificially intelligent architecture, at some point, in principle, understand and make breakthroughs in far-out, infamously difficult regimes of mathematics like string theory and theoretical physics? This research project thus began as a first step to prove to myself that indeed such artificially intelligent models are possible - perhaps even necessary.

I am grateful to JJ for his role in motivating this intellectual pursuit.

### Research and Engineering Overview

Research began by thoroughly investigating a variety of interesting mathematics datasets, including the [Google DeepMind mathematics dataset](https://huggingface.co/datasets/deepmind/math_dataset), the [Nvidia OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), etc., along with a number of compatible language models, including [FLAN-T5-large](https://huggingface.co/google/flan-t5-large), [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it), and more. My ultimate hope was to identify a (some) large dataset(s) along with a specific pretrained model architecture/checkpoint that seemed jointly compatible for finetuning to the rigorous regime of mathematical analysis. After much research, the combination of the DeepMind mathematics dataset with the pretrained Google FLAN-T5-large model seemed a very solid combination for finding eventually interesting and successful experimental results. Some guidance was provided by the [DeepMind publication](https://arxiv.org/abs/1904.01557) that introduced the DeepMind mathematics dataset.


### Model Cards

The model cards for CyberSolve-LinAlg are published on [my Hugging Face](https://huggingface.co/MarioBarbeque). See the list below for available links.
- Most recent version's model card: [CyberSolve-LinAlg-1.2](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.2)
- Initial version's modelcard: [CyberSolve-LinAlg-1.1](https://huggingface.co/MarioBarbeque/CyberSolve-LinAlg-1.1)

### GPU-optimized Inference

I have developed a dedicated application for doing Nvidia GPU-optimized inference with the most recent version of *CyberSolve-LinAlg* (v1.2) in the form of a Hugging Face space. 

The HF space makes use of the Python `streamlit` package, the Nvidia `apex` library, the full suite of the Hugging Face `transformers`, `tokenizers`, etc. ecosystem, and more to provide an extremely fast inference experience in querying the mathematical reasoning model through a simple UI. The finetuned versions of *CyberSolve* are no larger than about a few Gig, so they fit comfortably onto a single GPU. The dedicated inference application runs explicitly on an Nvidia T4.

This GPU-optimized inference of CyberSolve can be found here: [CyberSolve-LinAlg-1.2 inference](https://huggingface.co/spaces/MarioBarbeque/CyberSolveLinAlg1.2)

Try it out!

### Notebooks

This section contains links to the various notebooks used through the research process.

#### Preprocessing - [Preprocessing Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/preprocessing_and_inspection.ipynb)

The notebook here contains the intial steps to conduct some of this analysis while also preprocessing and cleaning up the dataset with robust considerations for training and evaluation. 

I took an interest in evaluating not only the exact correctness of the finetuned model, but also the partial correctness of the model's predictions. That is, did it somewhat accurately predict `21` when the answer was `21`, and so on. This subtelty turns out to be a difficult task; effort was dedicated in this first notebook to decipher a method that could scale across all predictions.

#### Benchmarking - [Benchmarking Notebook](https://github.com/johngrahamreynolds/CyberSolve-LinAlg/blob/main/benchmarking.ipynb)

This notebook contains the process of benchmarking the initial mathematical reasoning capability of the base FLAN-T5-large seq2seq model on the DeepMind mathematics evaluation dataset. 

As noted above, some effort was also given here to evaluate the FLAN-T5 model's partial correctness in solving linear equations. This analysis would, in principle, provide some added understanding of the model's true, underlying mathematical reasoning ability beyond just the exactness score. Given the difficult nature of this task, I make a few notes about the struggle with defining this process sysmetically and proceed preliminarily with analysis of the exactness alone. My interest to return to this analysis later on persists nonetheless.

#### Downsampled Training and Evaluation - [Downsampled Training Notebook]()

Before committing to the full, expensive process of finetuning on all 2M records in the DeepMind mathematics dataset's 1D linear algebra split, we conduct first a downsampled training to convince ourselves experimentally that indeed the FLAN-T5 model will generalize and domain-adapt well to the task of mathematical reasoning. 

#### Full Training and Evaluation - [Full Training and Eval Notebook]()

At last, we train our model on the entirety of the DeepMind mathematics dataset's 1D linear algebra split. 

### Conclusion, Intersting Future Endeavors

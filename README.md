# NetNet

My playground is a 125M scale Neural Network trained on data from the interNet.

## Goals:
* Portable code between Apple MPS and CUDA
* Train various types of neural networks from scratch
* Experiment with different architectures
* Test hypotheses about the nature of intelligence

## Setup

    source .venv/bin/activate
    pip install -r requirements.txt

    python download_data.py


## Base Architecture

A LLaMA-3.2 like architecture with 125M parameters.

* FineWeb-Edu dataset (HuggingFaceFW/fineweb-edu)
* Tokenizer: GPT-2 (GPT2TokenizerFast)
* RMSNorm (Root Mean Square Normalization)
* SwiGLU activation
* RoPE (Rotary Positional Embedding)

## License

MIT
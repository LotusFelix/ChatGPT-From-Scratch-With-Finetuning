# LLM From Scratch For Chatbot

This repo contains:
- A GPT‑style model built from scratch (attention, layers, blocks).
- Scripts to download OpenAI GPT‑2 weights and load into PyTorch.
- Fine‑tuning on Alpaca instruction data.
- Inference and evaluation examples.

## Getting Started

1. Clone:
   ```bash
   git clone https://github.com/LotusFelix/LLM-From-Scratch-For-Chatbot.git
   cd LLM-From-Scratch-For-Chatbot

pip install -r requirements.txt


from download import load_gpt2_checkpoint, load_params
from transformer.model import GPTModel

hparams, params = load_gpt2_checkpoint("gpt2/124M")
model = GPTModel(hparams)
load_params(model, params)

python train.py

python infer.py --input sample.json --output results.json --openai-key YOUR_KEY

"""
Generation and benchmarking scripts for GPTModel.
"""
import torch
import os
import json
import numpy as np
from transformer.model import GPTModel
from data.utils import format_input
import tiktoken
import openai

# Tokenizer
_tokenizer = tiktoken.get_encoding("gpt2")

def generate_text(model, prompt, max_new_tokens=100, top_k=None, temperature=1.0, eos_token=50256, device='cpu'):
    """Autoregressive text generation."""
    model.to(device).eval()
    tokens = _tokenizer.encode(prompt)
    x = torch.tensor(tokens, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config['context_length']:]
        logits = model(x_cond)
        next_logits = logits[:, -1, :]
        if top_k:
            vals, indices = torch.topk(next_logits, top_k)
            min_val = vals[:, -1].unsqueeze(1)
            next_logits = torch.where(next_logits < min_val, float('-inf'), next_logits)
        if temperature != 1.0:
            next_logits = next_logits / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == eos_token:
            break
        x = torch.cat([x, next_token], dim=1)
    return _tokenizer.decode(x.squeeze().tolist())


def benchmark_with_openai(data_json, output_json, openai_key, model_name='gpt-3.5-turbo', max_samples=20):
    """Evaluate model responses using OpenAI API scores."""
    openai.api_key = openai_key
    data = json.load(open(data_json))
    results = []
    for idx, entry in enumerate(data[:max_samples]):
        prompt = format_input(entry)
        # Generate response
        entry['model_response'] = generate_text(
            model, prompt,
            max_new_tokens=entry.get('max_new_tokens',256),
            top_k=entry.get('top_k',5),
            temperature=entry.get('temperature',0.7),
            eos_token=50256,
            device=next(model.parameters()).device
        )
        # Ask OpenAI to score
        chat = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role":"system","content":"You are a strict evaluator. Score from 0 to 100."},
                {"role":"user","content":(
                    f"Input: {prompt}
Correct: {entry['output']}
Response: {entry['model_response']}"
                )}
            ],
            temperature=0
        )
        entry['score'] = chat.choices[0].message.content.strip()
        results.append(entry)
    json.dump(results, open(output_json, 'w'), indent=2)
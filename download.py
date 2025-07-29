"""
Download and load GPT-2 pretrained weights into the model.
"""
import os
import json
import torch
import numpy as np
import urllib.request
from tqdm import tqdm
import tensorflow as tf


def download_file(url, dest, backup_url=None):
    """Download a file with progress bar."""
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            if os.path.exists(dest) and os.path.getsize(dest) == total:
                print(f"Skipping, up-to-date: {dest}")
                return
            desc = os.path.basename(dest)
            with open(dest, 'wb') as f, tqdm(total=total, unit='iB', unit_scale=True, desc=desc) as bar:
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception:
        if backup_url:
            print(f"Primary failed, trying backup.")
            download_file(backup_url, dest)
        else:
            raise


def load_gpt2_checkpoint(model_dir):
    """
    Load GPT-2 parameters from TensorFlow checkpoint in model_dir, return numpy params dict.
    """
    # Assuming checkpoint files present
    ckpt = tf.train.latest_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    hparams = json.load(open(os.path.join(model_dir, 'hparams.json')))
    # Load all variables
    params = {'blocks': [{} for _ in range(hparams['n_layer'])],
              'wte': None, 'wpe': None, 'ln_f': {}}
    for name, shape in tf.train.list_variables(ckpt):
        arr = np.squeeze(tf.train.load_variable(ckpt, name))
        parts = name.split('/')
        # Handle top-level embeddings and final norm
        if parts[0] == 'model' and parts[1] == 'wte':
            params['wte'] = arr
        elif parts[0] == 'model' and parts[1] == 'wpe':
            params['wpe'] = arr
        elif parts[0] == 'model' and parts[1] == 'ln_f':
            params['ln_f'][parts[-1]] = arr
        # Block parameters
        elif parts[0].startswith('h'):
            idx = int(parts[0][1:])
            subtree = params['blocks'][idx]
            # Nested dict insert
            target = subtree
            for key in parts[1:-1]:
                target = target.setdefault(key, {})
            target[parts[-1]] = arr
    return hparams, params


def load_params(model, params):
    """Assign numpy parameters into the PyTorch model."""
    from torch.nn import Parameter
    def assign_tensor(tparam, np_arr):
        if tparam.shape != np_arr.shape:
            raise ValueError(f"Shape mismatch: {tparam.shape} vs {np_arr.shape}")
        with torch.no_grad():
            tparam.copy_(torch.tensor(np_arr))
    # Embeddings
    assign_tensor(model.token_emb.weight, params['wte'])
    assign_tensor(model.pos_emb.weight,   params['wpe'])
    assign_tensor(model.head.weight,      params['wte'])
    # Final norm
    assign_tensor(model.ln_f.weight, params['ln_f']['g'])
    assign_tensor(model.ln_f.bias,   params['ln_f']['b'])
    # Blocks
    for i, block in enumerate(model.blocks):
        b = params['blocks'][i]
        # Attention QKV
        cw = b['attn']['c_attn']['w']
        cb = b['attn']['c_attn']['b']
        q, k, v = np.split(cw, 3, axis=-1)
        assign_tensor(block.attn.query_proj.weight, q.T)
        assign_tensor(block.attn.key_proj.weight,   k.T)
        assign_tensor(block.attn.value_proj.weight, v.T)
        assign_tensor(block.attn.query_proj.bias,   cb[:block.config['embed_dim']])
        assign_tensor(block.attn.key_proj.bias,     cb[block.config['embed_dim']:2*block.config['embed_dim']])
        assign_tensor(block.attn.value_proj.bias,   cb[2*block.config['embed_dim']:])
        # Attention output
        assign_tensor(block.attn.out_proj.weight, b['attn']['c_proj']['w'].T)
        assign_tensor(block.attn.out_proj.bias,   b['attn']['c_proj']['b'])
        # FeedForward
        assign_tensor(block.ff.fc1.weight, b['mlp']['c_fc']['w'].T)
        assign_tensor(block.ff.fc1.bias,   b['mlp']['c_fc']['b'])
        assign_tensor(block.ff.fc2.weight, b['mlp']['c_proj']['w'].T)
        assign_tensor(block.ff.fc2.bias,   b['mlp']['c_proj']['b'])
        # Layer norms
        assign_tensor(block.ln1.weight, b['ln_1']['g'])
        assign_tensor(block.ln1.bias,   b['ln_1']['b'])
        assign_tensor(block.ln2.weight, b['ln_2']['g'])
        assign_tensor(block.ln2.bias,   b['ln_2']['b'])
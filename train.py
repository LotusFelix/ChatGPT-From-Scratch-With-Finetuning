
"""
Training and evaluation loop for fine-tuning GPTModel on instruction data.
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformer.model import GPTModel
from data.dataset import InstructionDataset, custom_collate


def calculate_loss(model, x, y, device):
    """Compute cross-entropy loss on model predictions."""
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    x = x[:, -model.config['context_length']:]
    y = y[:, -model.config['context_length']:]
    logits = model(x)  # (batch, seq_len, vocab)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    return loss


def evaluate(model, dataloader, device, num_batches=None):
    """Compute average loss over a dataloader."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            if num_batches and idx >= num_batches:
                break
            loss = calculate_loss(model, x, y, device).item()
            total += loss
            count += 1
    model.train()
    return total / max(1, count)


def train(model, train_loader, val_loader, device,
          epochs=5, lr=1e-4, eval_freq=None, patience=3):
    """Fine-tune model with early stopping and track losses."""
    optimizer = AdamW(model.parameters(), lr=lr)
    best_val, no_improve = float('inf'), 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss(model, x, y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # Evaluation
        train_loss = evaluate(model, train_loader, device, eval_freq)
        val_loss = evaluate(model, val_loader, device, eval_freq)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        # Early stopping
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Stopping early at epoch {epoch}")
                break

    return train_losses, val_losses
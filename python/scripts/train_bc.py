import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jad import JadConfig, get_action_dims
from jad.env import get_observation_dim
from data import BCSequenceDataset, load_episodes
from models import LSTMPolicy


def train(
    data_dir: str = "data",
    jad_count: int = 1,
    healers_per_jad: int = 3,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 1,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    checkpoint_dir: str = "checkpoints",
    device: str = "auto",
):
    """Train LSTM policy via behavioral cloning"""
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Config
    config = JadConfig(jad_count=jad_count, healers_per_jad=healers_per_jad)
    obs_dim = get_observation_dim(config)
    action_dims = get_action_dims(config)

    print(f"Config: {jad_count} Jad(s), {healers_per_jad} healers per Jad")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dims: {action_dims} (MultiDiscrete)")

    # Load data
    data_path = Path(data_dir)
    print(f"\nLoading episodes from {data_path}...")
    episodes = load_episodes(data_path, config)
    print(f"Loaded {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No episodes found! Exiting.")
        return

    total_steps = sum(len(obs) for obs, _ in episodes)
    print(f"Total steps: {total_steps}")

    # Create dataset and dataloader
    dataset = BCSequenceDataset(episodes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Max sequence length: {dataset.max_seq_len}")

    # Create model
    model = LSTMPolicy(
        obs_dim=obs_dim,
        action_dims=action_dims,
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
    ).to(device)

    num_heads = len(action_dims)
    print(f"\nModel architecture:")
    print(f"  Features: {obs_dim} -> 64 -> 64")
    print(f"  LSTM: 64 -> {lstm_hidden_size} (layers={n_lstm_layers})")
    print(f"  Policy heads: {num_heads} heads with dims {action_dims}")
    print(f"  Value head: {lstm_hidden_size} -> 1")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    model_save_path = checkpoint_path / f"bc_{jad_count}jad_{healers_per_jad}heal.pt"

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_obs, batch_actions, batch_mask in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)  # (batch, seq_len, num_heads)
            batch_mask = batch_mask.to(device)

            # Forward pass - logits_list is list of (batch, seq_len, head_dim)
            logits_list, _, _ = model(batch_obs)

            # Compute loss for each head and sum
            total_loss = 0.0
            total_correct = 0
            mask_flat = batch_mask.view(-1)  # (batch * seq_len,)

            for head_idx, logits in enumerate(logits_list):
                head_dim = action_dims[head_idx]
                logits_flat = logits.view(-1, head_dim)  # (batch * seq_len, head_dim)
                actions_flat = batch_actions[:, :, head_idx].reshape(-1)  # (batch * seq_len,)

                loss_per_step = criterion(logits_flat, actions_flat)
                head_loss = (loss_per_step * mask_flat).sum() / mask_flat.sum()
                total_loss = total_loss + head_loss

                # Track accuracy per head
                preds = logits_flat.argmax(dim=-1)
                correct = ((preds == actions_flat) * mask_flat).sum().item()
                total_correct += correct

            # Average loss across heads
            loss = total_loss / num_heads

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics (accuracy is average across all heads)
            epoch_loss += loss.item() * mask_flat.sum().item()
            epoch_correct += total_correct
            epoch_total += mask_flat.sum().item() * num_heads

        # Epoch stats
        avg_loss = epoch_loss / epoch_total
        accuracy = epoch_correct / epoch_total

        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | Time: {elapsed_str}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'obs_dim': obs_dim,
                    'action_dims': action_dims,
                    'lstm_hidden_size': lstm_hidden_size,
                    'n_lstm_layers': n_lstm_layers,
                },
                'jad_config': {
                    'jad_count': jad_count,
                    'healers_per_jad': healers_per_jad,
                },
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
            }, model_save_path)
            print(f"  [Saved best model to {model_save_path}]")

    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM policy via behavioral cloning")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing episode JSON files",
    )
    parser.add_argument(
        "--jad-count",
        type=int,
        default=1,
        help="Number of Jads (default: 1)",
    )
    parser.add_argument(
        "--healers-per-jad",
        type=int,
        default=3,
        help="Number of healers per Jad (default: 3)",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=256,
        help="LSTM hidden layer size (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on (default: auto)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        jad_count=args.jad_count,
        healers_per_jad=args.healers_per_jad,
        lstm_hidden_size=args.lstm_hidden_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

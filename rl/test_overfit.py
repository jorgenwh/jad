"""
Test if the network can learn a simple pattern (overfit test).

If this fails, the architecture/optimization is broken before RL even starts.
"""

import torch
import torch.nn as nn
import numpy as np
from simple_agent import SimpleActorCritic
from model import ActorCritic
from observations import OBS_DIM


def create_synthetic_data(n_samples=300):
    """
    Create synthetic observations with clear action patterns:
    - jad_attack=none (one-hot [1,0,0]) → WAIT (action 0)
    - jad_attack=mage (one-hot [0,1,0]) → PRAY_MAGE (action 1)
    - jad_attack=range (one-hot [0,0,1]) → PRAY_RANGE (action 2)
    """
    observations = []
    actions = []

    for _ in range(n_samples):
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Random continuous features (normalized-ish)
        obs[0] = np.random.uniform(-1, 1)  # player_hp
        obs[1] = np.random.uniform(-1, 1)  # player_prayer
        obs[2] = np.random.uniform(-1, 1)  # jad_hp

        # Random active_prayer one-hot [3:6]
        active_idx = np.random.randint(0, 3)
        obs[3 + active_idx] = 1.0

        # Random jad_attack determines the correct action
        attack_type = np.random.randint(0, 3)  # 0=none, 1=mage, 2=range
        obs[6 + attack_type] = 1.0  # jad_attack one-hot [6:9]

        # Random restore_doses one-hot [9:14]
        doses = np.random.randint(0, 5)
        obs[9 + doses] = 1.0

        observations.append(obs)
        actions.append(attack_type)  # action = attack_type (0=wait, 1=pray_mage, 2=pray_range)

    return np.array(observations), np.array(actions)


def test_feedforward():
    """Test if feedforward network can overfit."""
    print("=" * 60)
    print("TEST 1: Feedforward Network (SimpleActorCritic)")
    print("=" * 60)

    obs, actions = create_synthetic_data(300)
    obs_tensor = torch.FloatTensor(obs)
    actions_tensor = torch.LongTensor(actions)

    model = SimpleActorCritic(obs_dim=OBS_DIM, action_dim=5, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(obs)} samples...")

    for epoch in range(100):
        logits, _ = model(obs_tensor)
        loss = criterion(logits, actions_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == actions_tensor).float().mean().item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Accuracy={accuracy*100:.1f}%")

    success = accuracy > 0.95
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'} ({accuracy*100:.1f}%)")
    return success


def test_lstm():
    """Test if LSTM network can overfit."""
    print()
    print("=" * 60)
    print("TEST 2: LSTM Network (ActorCritic) - actual training model")
    print("=" * 60)

    obs, actions = create_synthetic_data(300)
    obs_tensor = torch.FloatTensor(obs)
    actions_tensor = torch.LongTensor(actions)

    model = ActorCritic(obs_dim=OBS_DIM, action_dim=5, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(obs)} samples...")

    for epoch in range(100):
        # Process each sample individually (like during RL)
        all_logits = []
        hidden = model.init_hidden(batch_size=1)

        for i in range(len(obs_tensor)):
            single_obs = obs_tensor[i:i+1]  # (1, obs_dim)
            policy, _, hidden = model.forward(single_obs, hidden)
            all_logits.append(policy.logits)

        logits = torch.cat(all_logits, dim=0)  # (300, action_dim)
        loss = criterion(logits, actions_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == actions_tensor).float().mean().item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Accuracy={accuracy*100:.1f}%")

    success = accuracy > 0.95
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'} ({accuracy*100:.1f}%)")
    return success


def main():
    print()
    print("OVERFIT TEST: Can the networks learn a simple pattern?")
    print()
    print("Pattern: jad_attack type directly determines correct action")
    print("  - jad_attack=none  → action=WAIT (0)")
    print("  - jad_attack=mage  → action=PRAY_MAGE (1)")
    print("  - jad_attack=range → action=PRAY_RANGE (2)")
    print()

    ff_success = test_feedforward()
    lstm_success = test_lstm()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Feedforward: {'✓ PASS' if ff_success else '✗ FAIL'}")
    print(f"LSTM:        {'✓ PASS' if lstm_success else '✗ FAIL'}")

    if lstm_success:
        print("\nBoth networks can learn. Problem is in RL training loop.")
    elif ff_success and not lstm_success:
        print("\nLSTM cannot learn but feedforward can. LSTM is the problem.")
    else:
        print("\nNeither can learn. Fundamental architecture issue.")


if __name__ == "__main__":
    main()

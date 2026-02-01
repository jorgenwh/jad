# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### TypeScript (browser simulation)
```bash
npm run start      # Dev server at http://localhost:8080
npm run build      # Build to dist-browser/
npm run build:headless  # Build headless Node.js version to dist-headless/
npm run headless   # Run headless simulation
```

### Python (RL training) - requires `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"`
```bash
jad-train          # Train PPO agent (scripts/train_ppo.py)
jad-train-bc       # Train with behavioral cloning
jad-serve          # Serve trained agent via WebSocket
jad-test-env       # Test the gym environment
```

## Architecture

### Two-Layer Structure
The project separates browser-specific and shared logic:

- **src/core/** - Shared simulation logic (works in both browser and Node.js)
  - `jad-region.ts` - 27x27 arena, spawns player and Jads, manages healers
  - `jad.ts` - Jad mob with magic/range attacks, 3-tick projectile delay, healer spawning at 50% HP
  - `observation.ts` - Builds observation vectors from game state
  - `actions.ts` - Executes discrete actions (prayers, potions, targeting)
  - `rewards.ts` - Reward functions for RL training
- **src/browser/** - Browser entry point, rendering, agent WebSocket, recording
- **src/headless/** - Node.js execution with browser API mocks for training

### RL Training Pipeline
1. Python `EnvProcessWrapper` spawns headless TypeScript simulation as subprocess
2. Communication via stdin/stdout JSON protocol
3. `JadGymEnv` provides standard Gymnasium interface
4. Training uses RecurrentPPO (LSTM policy) from sb3-contrib

### Browser Modes
- **Normal**: `http://localhost:8080` - Manual play
- **Agent**: `?agent=true` - WebSocket-controlled by Python agent server
- **Recording**: `?record=true` - Captures gameplay for behavioral cloning
- **Multi-Jad**: `?jads=N&healers=H` - Configure 1-6 Jads with 0-5 healers each

### Action Space
Actions are discrete integers:
- 0: DO_NOTHING
- 1..N: Attack Jad 1 through N
- N+1..N+N*H: Attack healers (encoded by Jad index and healer index)
- Remaining: Toggle prayers (Melee/Range/Magic/Rigour), drink potions (Bastion/Restore/Brew)

### Key Game Mechanics
- **Tick system**: 600ms game ticks
- **Jad attacks**: 50/50 magic/range, 3-tick projectile delay
- **Prayer protection**: Must be active when projectile hits (not on attack start)
- **Healers**: Spawn when Jad drops below 50% HP, heal Jad unless aggro'd onto player

### osrs-sdk Dependency
Core game systems from `osrs-sdk`: `World`, `Region`, `Player`, `Mob`, `Unit`, weapons, projectiles, rendering

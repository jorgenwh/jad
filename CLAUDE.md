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
The environment uses a **MultiDiscrete** action space with 4 heads, allowing simultaneous actions per tick (e.g., switch prayer AND attack AND drink potion).

| Head | Name | Size | Values |
|------|------|------|--------|
| 0 | `protection_prayer` | 4 | 0=no-op, 1=protect_mage, 2=protect_range, 3=protect_melee |
| 1 | `offensive_prayer` | 2 | 0=no-op, 1=toggle_rigour |
| 2 | `potion` | 4 | 0=none, 1=bastion, 2=super_restore, 3=sara_brew |
| 3 | `target` | 1+N+N*H | 0=no-op, 1..N=jad, N+1..=healer |

**Action semantics:**
- Protection prayer: toggle semantics (click to toggle prayer on/off)
- Offensive prayer: toggle semantics (click to toggle Rigour)
- Potion: drink specified potion
- Target: switch to specified target (no-op keeps current)

**Example for 2 Jads, 3 healers each (N=2, H=3):**
```python
spaces.MultiDiscrete([4, 2, 4, 9])  # 4 heads

# Sample action: [1, 0, 2, 3]
# - Toggle protect mage
# - No offensive prayer change
# - Drink super restore
# - Attack healer 1 of Jad 1
```

Key files:
- `src/core/actions.ts` - `executeAction()`, `getActionSpaceDims()`
- `python/jad/actions.py` - `get_action_dims()`

### Action Masking
Action masking prevents the agent from selecting invalid actions during training and inference. Masks are **per-head boolean arrays** where `true` = valid action.

**Per-head mask rules** (implemented in `buildValidActionMask()` in `src/core/actions.ts`):

| Head | Mask Logic |
|------|------------|
| 0 (protection) | `[true, true, true, true]` - always valid |
| 1 (offensive) | `[true, true]` - always valid |
| 2 (potion) | `[true, bastion>0, restore>0, brew>0]` |
| 3 (target) | `[true, jad1_alive, jad2_alive, h1.1_alive, ...]` |

**Data flow**:
1. TypeScript `HeadlessEnv.step()` calls `buildValidActionMask()` after each tick
2. Mask included in `StepResult.valid_action_mask` (array of boolean arrays)
3. Sent to Python via JSON stdin/stdout protocol
4. `JadGymEnv.action_masks()` returns tuple of numpy arrays (one per head)

**Gym environment interface**:
```python
env = JadGymEnv(config)
obs, info = env.reset()
masks = env.action_masks()  # tuple of np.ndarray, one per head
# masks[0].shape = (4,)  # protection prayer
# masks[1].shape = (2,)  # offensive prayer
# masks[2].shape = (4,)  # potion
# masks[3].shape = (1+N+N*H,)  # target
```

### Key Game Mechanics
- **Tick system**: 600ms game ticks
- **Jad attacks**: 50/50 magic/range, 3-tick projectile delay
- **Prayer protection**: Must be active when projectile hits (not on attack start)
- **Healers**: Spawn when Jad drops below 50% HP, heal Jad unless aggro'd onto player

### osrs-sdk Dependency
Core game systems from `osrs-sdk`: `World`, `Region`, `Player`, `Mob`, `Unit`, weapons, projectiles, rendering

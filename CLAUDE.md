# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jad Trainer is an OSRS (Old School RuneScape) browser-based simulation for practicing the Jad fight. It uses the `osrs-sdk` library to render a 3D game world with a player fighting the Jal-Tok-Jad boss.

## Commands

- `npm run start` - Start development server with hot reload (opens at http://localhost:8080)
- `npm run build` - Build for development to `dist-browser/`

## Architecture

### Entry Point Flow
`src/index.ts` initializes the game:
1. Creates `JadRegion` and `World` instances
2. Sets up player with loadout from `loadout.ts`
3. Waits for both images and 3D assets to load before starting the world tick

### Core Files

- **src/jad-region.ts**: Defines the 20x20 arena region, spawns the player and Jad mob, handles boundary blockers and floor rendering
- **src/jad.ts**: The Jad mob implementation with custom magic/range weapons that use delayed attacks with projectile timing. Key mechanics:
  - 50/50 random attack style (magic vs range) via `attackStyleForNewAttack()`
  - 3-tick projectile delay (`JAD_PROJECTILE_DELAY`)
  - Prayer protection checks on hit (not on attack start)
- **src/loadout.ts**: Player equipment configuration (Twisted Bow setup with Masori gear)

### osrs-sdk Dependency
The project relies heavily on `osrs-sdk` for:
- Core game systems: `World`, `Region`, `Player`, `Mob`, `Unit`
- Combat: `MagicWeapon`, `RangedWeapon`, `Projectile`, `AttackBonuses`
- Rendering: `Viewport`, `GLTFModel`, `Assets`
- UI: `Trainer`, `MapController`, `ImageLoader`

/**
 * Headless environment wrapper for RL training.
 * Supports 1-6 Jads with per-Jad healers.
 */

import { World, Region, Player, Potion, Settings, Trainer } from 'osrs-sdk';
import { Jad } from '../jad';
import { JadRegion, HealerAggro } from '../jad-region';
import { YtHurKot } from '../healer';
import { JadConfig, DEFAULT_CONFIG, getActionCount } from '../config';

// Initialize settings from storage (uses mock localStorage)
Settings.readFromStorage();

/**
 * Get action count for a given configuration.
 * Actions: DO_NOTHING + N*AGGRO_JAD + N*3*AGGRO_HEALER + 7 prayers/potions
 */
export { getActionCount };

/**
 * Action structure for N Jads:
 * - 0: DO_NOTHING
 * - 1..N: AGGRO_JAD_1 through AGGRO_JAD_N
 * - N+1..N+3N: AGGRO_HEALER (encoded as jadIndex * healersPerJad + healerIndex)
 * - N+3N+1: TOGGLE_PROTECT_MELEE
 * - N+3N+2: TOGGLE_PROTECT_MISSILES
 * - N+3N+3: TOGGLE_PROTECT_MAGIC
 * - N+3N+4: TOGGLE_RIGOUR
 * - N+3N+5: DRINK_BASTION
 * - N+3N+6: DRINK_SUPER_RESTORE
 * - N+3N+7: DRINK_SARA_BREW
 */

// Single Jad state in observation
export interface JadState {
  hp: number;
  attack: number;  // 0=none, 1=mage, 2=range, 3=melee
  x: number;
  y: number;
  alive: boolean;
}

// Single healer state in observation
export interface HealerState {
  hp: number;
  x: number;
  y: number;
  aggro: number;  // HealerAggro enum: 0=not_present, 1=jad, 2=player
}

export interface JadObservation {
  // Config
  jad_count: number;
  healers_per_jad: number;

  // Player state
  player_hp: number;
  player_prayer: number;
  player_ranged: number;    // Current ranged stat (1-118)
  player_defence: number;   // Current defence stat (1-118)
  player_x: number;         // Player x position (0-19)
  player_y: number;         // Player y position (0-19)
  player_aggro: number;     // 0=none, 1..N=jad_N, N+1..=healer

  // Prayer state
  active_prayer: number;    // 0=none, 1=mage, 2=range, 3=melee
  rigour_active: boolean;

  // Inventory
  bastion_doses: number;
  sara_brew_doses: number;
  super_restore_doses: number;

  // Dynamic Jad state array
  jads: JadState[];

  // Dynamic healer state array (flattened: [jad0_healer0, jad0_healer1, ..., jad1_healer0, ...])
  healers: HealerState[];

  // Whether any healers have spawned
  healers_spawned: boolean;

  // Starting doses for normalization
  starting_bastion_doses: number;
  starting_sara_brew_doses: number;
  starting_super_restore_doses: number;
}

export interface StepResult {
  observation: JadObservation;
  terminated: boolean;
}

export interface ResetResult {
  observation: JadObservation;
}

// Per-Jad attack tracking state
interface JadAttackState {
  attack: number;  // 0=none, 1=mage, 2=range, 3=melee
  ticksRemaining: number;
  prevAttackDelay: number;
}

export class HeadlessEnv {
  private world: World;
  private region: Region;
  private player!: Player;
  private config: JadConfig;
  private createRegion: (config: JadConfig) => Region;

  // Track attack state per Jad
  private jadAttackStates: JadAttackState[] = [];

  // Track starting potion doses for normalization
  private startingBastionDoses: number = 0;
  private startingSaraBrewDoses: number = 0;
  private startingSuperRestoreDoses: number = 0;

  constructor(createRegion: (config: JadConfig) => Region, config: JadConfig = DEFAULT_CONFIG) {
    this.config = config;
    this.createRegion = createRegion;
    this.world = new World();
    this.region = createRegion(config);
    this.initializeRegion();
  }

  private initializeRegion(): void {
    this.region.world = this.world;
    this.world.addRegion(this.region);

    // Initialize region and get player
    const result = (this.region as { initialiseRegion(): { player: Player } }).initialiseRegion();
    this.player = result.player;

    // Set player in Trainer (used by osrs-sdk internals)
    Trainer.setPlayer(this.player);

    // Initialize per-Jad attack tracking
    this.jadAttackStates = [];
    const jadRegion = this.region as JadRegion;
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = jadRegion.getJad(i);
      this.jadAttackStates.push({
        attack: 0,
        ticksRemaining: 0,
        prevAttackDelay: jad?.attackDelay ?? 0,
      });
    }

    // Capture starting potion doses for normalization
    this.captureStartingDoses();
  }

  private captureStartingDoses(): void {
    let bastionDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;

    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('bastion')) {
            bastionDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
          } else if (itemName.includes('restore')) {
            superRestoreDoses += item.doses;
          }
        }
      }
    }

    this.startingBastionDoses = bastionDoses;
    this.startingSaraBrewDoses = saraBrewDoses;
    this.startingSuperRestoreDoses = superRestoreDoses;
  }

  reset(): ResetResult {
    // Clear and recreate
    this.world = new World();
    this.region = this.createRegion(this.config);
    this.initializeRegion();

    return {
      observation: this.getObservation(),
    };
  }

  step(action: number): StepResult {
    // Execute action before tick
    this.executeAction(action);

    // Update Jad attack tracking before tick
    this.updateJadAttackTracking();

    // Advance simulation by one tick
    this.world.tickWorld(1);

    // Get observation after tick
    const observation = this.getObservation();

    // Check termination - player dead or ALL Jads dead
    const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
    const jadRegion = this.region as JadRegion;
    let allJadsDead = true;
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = jadRegion.getJad(i);
      if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
        allJadsDead = false;
        break;
      }
    }
    const terminated = playerDead || allJadsDead;

    return {
      observation,
      terminated,
    };
  }

  /**
   * Execute action based on dynamic action space.
   * Action structure:
   * - 0: DO_NOTHING
   * - 1..N: AGGRO_JAD_1 through AGGRO_JAD_N
   * - N+1..N+N*H: AGGRO_HEALER (encoded)
   * - N+N*H+1..N+N*H+7: prayers/potions
   */
  private executeAction(action: number): void {
    const jadRegion = this.region as JadRegion;
    const numJads = this.config.jadCount;
    const healersPerJad = this.config.healersPerJad;
    const totalHealers = numJads * healersPerJad;

    // Action 0: DO_NOTHING
    if (action === 0) {
      return;
    }

    // Actions 1..N: AGGRO_JAD_1 through AGGRO_JAD_N
    if (action >= 1 && action <= numJads) {
      const jadIndex = action - 1;
      const jad = jadRegion.getJad(jadIndex);
      if (jad) {
        console.log(`[AGGRO] Targeting Jad ${jadIndex} (HP: ${jad.currentStats.hitpoint})`);
        this.player.setAggro(jad);
      } else {
        // Debug: check why jad is null
        const rawJad = jadRegion['_jads'][jadIndex];
        if (!rawJad) {
          console.log(`[AGGRO] Jad ${jadIndex} not found (never spawned)`);
        } else {
          console.log(`[AGGRO] Jad ${jadIndex} is dead (HP: ${rawJad.currentStats.hitpoint}, dying: ${rawJad.dying})`);
        }
      }
      return;
    }

    // Actions N+1..N+N*H: AGGRO_HEALER
    const healerActionStart = numJads + 1;
    const healerActionEnd = numJads + totalHealers;
    if (action >= healerActionStart && action <= healerActionEnd) {
      const healerActionIndex = action - healerActionStart;
      const jadIndex = Math.floor(healerActionIndex / healersPerJad);
      const healerIndex = healerActionIndex % healersPerJad;
      const healer = jadRegion.getHealer(jadIndex, healerIndex);
      if (healer) {
        console.log(`[AGGRO] Targeting Healer [${jadIndex}][${healerIndex}] (HP: ${healer.currentStats.hitpoint})`);
        this.player.setAggro(healer);
      } else {
        // Debug: check why healer is null
        const healerTuple = jadRegion['_healers'].get(jadIndex);
        const rawHealer = healerTuple?.[healerIndex];
        if (!rawHealer) {
          console.log(`[AGGRO] Healer [${jadIndex}][${healerIndex}] not spawned yet`);
        } else {
          console.log(`[AGGRO] Healer [${jadIndex}][${healerIndex}] is dead (HP: ${rawHealer.currentStats.hitpoint}, dying: ${rawHealer.dying})`);
        }
      }
      return;
    }

    // Prayer/potion actions (offset by aggro actions)
    const prayerPotionActionStart = numJads + totalHealers + 1;
    const relativeAction = action - prayerPotionActionStart;

    switch (relativeAction) {
      case 0: // TOGGLE_PROTECT_MELEE
        this.togglePrayer('Protect from Melee');
        break;
      case 1: // TOGGLE_PROTECT_MISSILES
        this.togglePrayer('Protect from Range');
        break;
      case 2: // TOGGLE_PROTECT_MAGIC
        this.togglePrayer('Protect from Magic');
        break;
      case 3: // TOGGLE_RIGOUR
        this.togglePrayer('Rigour');
        break;
      case 4: // DRINK_BASTION
        this.drinkPotion('bastion');
        break;
      case 5: // DRINK_SUPER_RESTORE
        this.drinkPotion('restore');
        break;
      case 6: // DRINK_SARA_BREW
        this.drinkPotion('saradomin brew');
        break;
    }
  }

  private togglePrayer(prayerName: string): void {
    const prayerController = this.player.prayerController;
    if (!prayerController) return;

    const targetPrayer = prayerController.findPrayerByName(prayerName);
    if (targetPrayer) {
      if (targetPrayer.isActive) {
        targetPrayer.deactivate();
      } else {
        // Can't activate prayers with 0 prayer points
        if ((this.player.currentStats?.prayer ?? 0) > 0) {
          targetPrayer.activate(this.player);
        }
      }
    }
  }

  private drinkPotion(potionType: string): void {
    if (!this.player || !this.player.inventory) return;
    const inventory = this.player.inventory;

    for (const item of inventory) {
      if (item && item instanceof Potion && item.doses > 0) {
        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes(potionType)) {
          item.inventoryLeftClick(this.player);
          break;
        }
      }
    }
  }

  private updateJadAttackTracking(): void {
    const jadRegion = this.region as JadRegion;

    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = jadRegion.getJad(i);
      if (!jad) continue;

      const state = this.jadAttackStates[i];
      const currentDelay = jad.attackDelay;

      // Detect actual attack: attackDelay was low (0-1), now it's high (just reset after attacking)
      if (state.prevAttackDelay <= 1 && currentDelay > 1) {
        const style = jad.attackStyle;
        if (style === 'magic') {
          state.attack = 1;
          state.ticksRemaining = 4;
        } else if (style === 'range') {
          state.attack = 2;
          state.ticksRemaining = 4;
        } else {
          state.attack = 3;
          state.ticksRemaining = 2;
        }
      }

      // Decrement visibility timer
      if (state.ticksRemaining > 0) {
        state.ticksRemaining--;
        if (state.ticksRemaining === 0) {
          state.attack = 0;
        }
      }

      state.prevAttackDelay = currentDelay;
    }
  }

  private getObservation(): JadObservation {
    const jadRegion = this.region as JadRegion;

    // Get active prayer
    let activePrayer = 0;
    let rigourActive = false;
    const prayerController = this.player.prayerController;
    if (prayerController) {
      const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
      const rangePrayer = prayerController.findPrayerByName('Protect from Range');
      const meleePrayer = prayerController.findPrayerByName('Protect from Melee');
      const rigourPrayer = prayerController.findPrayerByName('Rigour');

      if (magicPrayer?.isActive) {
        activePrayer = 1;
      } else if (rangePrayer?.isActive) {
        activePrayer = 2;
      } else if (meleePrayer?.isActive) {
        activePrayer = 3;
      }

      rigourActive = rigourPrayer?.isActive ?? false;
    }

    // Count current potion doses
    let bastionDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('bastion')) {
            bastionDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
          } else if (itemName.includes('restore')) {
            superRestoreDoses += item.doses;
          }
        }
      }
    }

    // Determine player aggro target
    // 0=none, 1..N=jad_N, N+1..N+N*H=healer (encoded)
    let playerAggro = 0;
    const playerAggroTarget = this.player.aggro;
    if (playerAggroTarget) {
      // Check if aggro is one of the Jads
      for (let i = 0; i < this.config.jadCount; i++) {
        const jad = jadRegion.getJad(i);
        if (jad && playerAggroTarget === jad) {
          playerAggro = i + 1;  // 1-indexed Jad
          break;
        }
      }

      // If not a Jad, check if it's a healer
      if (playerAggro === 0) {
        const numJads = this.config.jadCount;
        const healersPerJad = this.config.healersPerJad;
        for (let jadIdx = 0; jadIdx < numJads; jadIdx++) {
          for (let healerIdx = 0; healerIdx < healersPerJad; healerIdx++) {
            const healer = jadRegion.getHealer(jadIdx, healerIdx);
            if (healer && playerAggroTarget === healer) {
              // Encode as: numJads + jadIdx * healersPerJad + healerIdx + 1
              playerAggro = numJads + jadIdx * healersPerJad + healerIdx + 1;
              break;
            }
          }
          if (playerAggro !== 0) break;
        }
      }
    }

    // Build Jad state array
    const jads: JadState[] = [];
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = jadRegion.getJad(i);
      const attackState = this.jadAttackStates[i];
      if (jad) {
        jads.push({
          hp: jad.currentStats?.hitpoint ?? 0,
          attack: attackState?.attack ?? 0,
          x: jad.location.x,
          y: jad.location.y,
          alive: jad.dying === -1 && (jad.currentStats?.hitpoint ?? 0) > 0,
        });
      } else {
        jads.push({ hp: 0, attack: 0, x: 0, y: 0, alive: false });
      }
    }

    // Build healer state array (flattened)
    const healers: HealerState[] = [];
    let healersSpawned = false;
    for (let jadIdx = 0; jadIdx < this.config.jadCount; jadIdx++) {
      for (let healerIdx = 0; healerIdx < this.config.healersPerJad; healerIdx++) {
        const healer = jadRegion.getHealer(jadIdx, healerIdx);
        if (healer) {
          healersSpawned = true;
          healers.push({
            hp: healer.currentStats?.hitpoint ?? 0,
            x: healer.location.x,
            y: healer.location.y,
            aggro: jadRegion.getHealerAggro(jadIdx, healerIdx),
          });
        } else {
          healers.push({
            hp: 0,
            x: 0,
            y: 0,
            aggro: HealerAggro.NOT_PRESENT,
          });
        }
      }
    }

    return {
      // Config
      jad_count: this.config.jadCount,
      healers_per_jad: this.config.healersPerJad,

      // Player state
      player_hp: this.player?.currentStats?.hitpoint ?? 0,
      player_prayer: this.player?.currentStats?.prayer ?? 0,
      player_ranged: this.player?.currentStats?.range ?? 99,
      player_defence: this.player?.currentStats?.defence ?? 99,
      player_x: this.player?.location?.x ?? 0,
      player_y: this.player?.location?.y ?? 0,
      player_aggro: playerAggro,

      // Prayer state
      active_prayer: activePrayer,
      rigour_active: rigourActive,

      // Inventory
      bastion_doses: bastionDoses,
      sara_brew_doses: saraBrewDoses,
      super_restore_doses: superRestoreDoses,

      // Jad state
      jads: jads,

      // Healer state
      healers: healers,
      healers_spawned: healersSpawned,

      // Starting doses for normalization
      starting_bastion_doses: this.startingBastionDoses,
      starting_sara_brew_doses: this.startingSaraBrewDoses,
      starting_super_restore_doses: this.startingSuperRestoreDoses,
    };
  }
}

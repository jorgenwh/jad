/**
 * Headless environment wrapper for RL training.
 */

import { World, Region, Player, Potion, Settings, Trainer } from 'osrs-sdk';
import { Jad } from '../jad';
import { JadRegion, HealerAggro } from '../jad-region';
import { YtHurKot } from '../healer';

// Initialize settings from storage (uses mock localStorage)
Settings.readFromStorage();

// Action enum (12 discrete actions)
export enum JadAction {
  DO_NOTHING = 0,
  AGGRO_JAD = 1,
  AGGRO_HEALER_1 = 2,
  AGGRO_HEALER_2 = 3,
  AGGRO_HEALER_3 = 4,
  TOGGLE_PROTECT_MELEE = 5,
  TOGGLE_PROTECT_MISSILES = 6,
  TOGGLE_PROTECT_MAGIC = 7,
  TOGGLE_PIETY = 8,
  DRINK_SUPER_COMBAT = 9,
  DRINK_SUPER_RESTORE = 10,
  DRINK_SARA_BREW = 11,
}

// Player aggro target for observation
export enum PlayerAggro {
  NONE = 0,
  JAD = 1,
  HEALER_1 = 2,
  HEALER_2 = 3,
  HEALER_3 = 4,
}

export interface JadObservation {
  // Player state
  player_hp: number;
  player_prayer: number;
  player_attack: number;    // Current attack stat (1-118)
  player_strength: number;  // Current strength stat (1-118)
  player_defence: number;   // Current defence stat (1-118)
  player_x: number;         // Player x position (0-19)
  player_y: number;         // Player y position (0-19)
  player_aggro: number;     // PlayerAggro enum: 0=none, 1=jad, 2-4=healer 1-3

  // Prayer state
  active_prayer: number;    // 0=none, 1=mage, 2=range, 3=melee
  piety_active: boolean;

  // Inventory
  super_combat_doses: number;
  sara_brew_doses: number;
  super_restore_doses: number;

  // Jad state
  jad_hp: number;
  jad_attack: number;       // 0=none, 1=mage, 2=range, 3=melee
  jad_x: number;            // Jad x position (0-19)
  jad_y: number;            // Jad y position (0-19)

  // Healer state
  healers_spawned: boolean;
  healer_1_hp: number;      // 0 if not present
  healer_1_x: number;
  healer_1_y: number;
  healer_1_aggro: number;   // HealerAggro enum: 0=not_present, 1=jad, 2=player
  healer_2_hp: number;
  healer_2_x: number;
  healer_2_y: number;
  healer_2_aggro: number;
  healer_3_hp: number;
  healer_3_x: number;
  healer_3_y: number;
  healer_3_aggro: number;

  // Starting doses for normalization
  starting_super_combat_doses: number;
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

export class HeadlessEnv {
  private world: World;
  private region: Region;
  private player!: Player;
  private jad!: Jad;
  private createRegion: () => Region;

  // Track Jad's attack for observation
  private currentJadAttack: number = 0; // 0=none, 1=mage, 2=range, 3=melee
  private attackTicksRemaining: number = 0;
  private prevJadAttackDelay: number = 0; // To detect when Jad actually attacks

  // Track starting potion doses for normalization
  private startingSuperCombatDoses: number = 0;
  private startingSaraBrewDoses: number = 0;
  private startingSuperRestoreDoses: number = 0;

  constructor(createRegion: () => Region) {
    this.createRegion = createRegion;
    this.world = new World();
    this.region = createRegion();
    this.initializeRegion();
  }

  private initializeRegion(): void {
    this.region.world = this.world;
    this.world.addRegion(this.region);

    // Initialize region and get player
    const result = (this.region as { initialiseRegion(): { player: Player } }).initialiseRegion();
    this.player = result.player;

    // Get Jad from the region's jad getter
    this.jad = (this.region as JadRegion).jad;

    // Set player in Trainer (used by osrs-sdk internals)
    Trainer.setPlayer(this.player);

    // Initialize attack tracking
    this.currentJadAttack = 0;
    this.attackTicksRemaining = 0;
    this.prevJadAttackDelay = this.jad.attackDelay;

    // Capture starting potion doses for normalization
    this.captureStartingDoses();
  }

  private captureStartingDoses(): void {
    let superCombatDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;

    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('super combat')) {
            superCombatDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
          } else if (itemName.includes('restore')) {
            superRestoreDoses += item.doses;
          }
        }
      }
    }

    this.startingSuperCombatDoses = superCombatDoses;
    this.startingSaraBrewDoses = saraBrewDoses;
    this.startingSuperRestoreDoses = superRestoreDoses;
  }

  reset(): ResetResult {
    // Clear and recreate
    this.world = new World();
    this.region = this.createRegion();
    this.initializeRegion();

    return {
      observation: this.getObservation(),
    };
  }

  step(action: number): StepResult {
    // Execute action before tick
    this.executeAction(action);

    // Observe Jad's attack style before tick (for tracking)
    this.updateJadAttackTracking();

    // Advance simulation by one tick
    this.world.tickWorld(1);

    // Get observation after tick
    const observation = this.getObservation();

    // Check termination
    const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
    const jadDead = this.jad.dying > 0 || this.jad.currentStats.hitpoint <= 0;
    const terminated = playerDead || jadDead;

    return {
      observation,
      terminated,
    };
  }

  private executeAction(action: number): void {
    const jadRegion = this.region as JadRegion;

    switch (action) {
      case JadAction.DO_NOTHING:
        // Do nothing
        break;

      case JadAction.AGGRO_JAD:
        this.player.setAggro(this.jad);
        break;

      case JadAction.AGGRO_HEALER_1:
        const healer1 = jadRegion.getHealer(0);
        if (healer1) this.player.setAggro(healer1);
        break;

      case JadAction.AGGRO_HEALER_2:
        const healer2 = jadRegion.getHealer(1);
        if (healer2) this.player.setAggro(healer2);
        break;

      case JadAction.AGGRO_HEALER_3:
        const healer3 = jadRegion.getHealer(2);
        if (healer3) this.player.setAggro(healer3);
        break;

      case JadAction.TOGGLE_PROTECT_MELEE:
        this.togglePrayer('Protect from Melee');
        break;

      case JadAction.TOGGLE_PROTECT_MISSILES:
        this.togglePrayer('Protect from Range');
        break;

      case JadAction.TOGGLE_PROTECT_MAGIC:
        this.togglePrayer('Protect from Magic');
        break;

      case JadAction.TOGGLE_PIETY:
        this.togglePrayer('Piety');
        break;

      case JadAction.DRINK_SUPER_COMBAT:
        this.drinkPotion('super combat');
        break;

      case JadAction.DRINK_SUPER_RESTORE:
        this.drinkPotion('restore');
        break;

      case JadAction.DRINK_SARA_BREW:
        this.drinkPotion('saradomin brew');
        break;
    }
  }

  private togglePrayer(prayerName: string): void {
    const prayerController = this.player.prayerController;
    if (!prayerController) return;

    // Find the prayer by name
    const targetPrayer = prayerController.findPrayerByName(prayerName);

    if (targetPrayer) {
      if (targetPrayer.isActive) {
        // Deactivate if already active
        targetPrayer.deactivate();
      } else {
        // Activate if not active - this auto-deactivates conflicting overhead prayers
        targetPrayer.activate(this.player);
      }
    }
  }

  private drinkPotion(potionType: string): void {
    // Find a potion in inventory and drink it
    if (!this.player || !this.player.inventory) return;
    const inventory = this.player.inventory;

    for (const item of inventory) {
      if (item && item instanceof Potion && item.doses > 0) {
        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes(potionType)) {
          // Use inventoryLeftClick to properly go through eating system
          // This decrements doses and queues the potion effect for next tick
          item.inventoryLeftClick(this.player);
          break;
        }
      }
    }
  }

  private updateJadAttackTracking(): void {
    // Track Jad's attack style
    // The attack style is visible when Jad attacks and persists until damage lands

    const currentDelay = this.jad.attackDelay;

    // Detect actual attack: attackDelay was low (0-1), now it's high (just reset after attacking)
    if (this.prevJadAttackDelay <= 1 && currentDelay > 1) {
      // Jad just attacked - capture the attack style NOW
      const style = this.jad.attackStyle;
      if (style === 'magic') {
        this.currentJadAttack = 1;
        this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
      } else if (style === 'range') {
        this.currentJadAttack = 2;
        this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
      } else {
        // Melee attack - style is 'stab' (from canMeleeIfClose)
        // Melee is instant, no projectile delay
        // Use 2 so after immediate decrement it's 1 (visible for 1 tick)
        this.currentJadAttack = 3;
        this.attackTicksRemaining = 2;
      }
    }

    // Decrement visibility timer
    if (this.attackTicksRemaining > 0) {
      this.attackTicksRemaining--;
      if (this.attackTicksRemaining === 0) {
        this.currentJadAttack = 0;
      }
    }

    this.prevJadAttackDelay = currentDelay;
  }

  private getObservation(): JadObservation {
    const jadRegion = this.region as JadRegion;

    // Get active prayer (0=none, 1=mage, 2=range, 3=melee)
    let activePrayer = 0;
    let pietyActive = false;
    const prayerController = this.player.prayerController;
    if (prayerController) {
      const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
      const rangePrayer = prayerController.findPrayerByName('Protect from Range');
      const meleePrayer = prayerController.findPrayerByName('Protect from Melee');
      const pietyPrayer = prayerController.findPrayerByName('Piety');

      if (magicPrayer?.isActive) {
        activePrayer = 1;
      } else if (rangePrayer?.isActive) {
        activePrayer = 2;
      } else if (meleePrayer?.isActive) {
        activePrayer = 3;
      }

      pietyActive = pietyPrayer?.isActive ?? false;
    }

    // Count current potion doses
    let superCombatDoses = 0;
    let saraBrewDoses = 0;
    let superRestoreDoses = 0;
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('super combat')) {
            superCombatDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
          } else if (itemName.includes('restore')) {
            superRestoreDoses += item.doses;
          }
        }
      }
    }

    // Determine player aggro target
    let playerAggro = PlayerAggro.NONE;
    if (this.player.aggro === this.jad) {
      playerAggro = PlayerAggro.JAD;
    } else if (this.player.aggro) {
      // Check if aggro is one of the healers
      for (let i = 0; i < 3; i++) {
        const healer = jadRegion.getHealer(i);
        if (healer && this.player.aggro === healer) {
          playerAggro = PlayerAggro.HEALER_1 + i; // HEALER_1=2, HEALER_2=3, HEALER_3=4
          break;
        }
      }
    }

    // Get healer data
    const getHealerData = (index: number) => {
      const healer = jadRegion.getHealer(index);
      if (healer) {
        return {
          hp: healer.currentStats?.hitpoint ?? 0,
          x: healer.location.x,
          y: healer.location.y,
          aggro: jadRegion.getHealerAggro(index),
        };
      }
      return { hp: 0, x: 0, y: 0, aggro: HealerAggro.NOT_PRESENT };
    };

    const healer1 = getHealerData(0);
    const healer2 = getHealerData(1);
    const healer3 = getHealerData(2);

    return {
      // Player state
      player_hp: this.player?.currentStats?.hitpoint ?? 0,
      player_prayer: this.player?.currentStats?.prayer ?? 0,
      player_attack: this.player?.currentStats?.attack ?? 99,
      player_strength: this.player?.currentStats?.strength ?? 99,
      player_defence: this.player?.currentStats?.defence ?? 99,
      player_x: this.player?.location?.x ?? 0,
      player_y: this.player?.location?.y ?? 0,
      player_aggro: playerAggro,

      // Prayer state
      active_prayer: activePrayer,
      piety_active: pietyActive,

      // Inventory
      super_combat_doses: superCombatDoses,
      sara_brew_doses: saraBrewDoses,
      super_restore_doses: superRestoreDoses,

      // Jad state
      jad_hp: this.jad?.currentStats?.hitpoint ?? 0,
      jad_attack: this.currentJadAttack,
      jad_x: this.jad?.location?.x ?? 0,
      jad_y: this.jad?.location?.y ?? 0,

      // Healer state
      healers_spawned: jadRegion.healersSpawned,
      healer_1_hp: healer1.hp,
      healer_1_x: healer1.x,
      healer_1_y: healer1.y,
      healer_1_aggro: healer1.aggro,
      healer_2_hp: healer2.hp,
      healer_2_x: healer2.x,
      healer_2_y: healer2.y,
      healer_2_aggro: healer2.aggro,
      healer_3_hp: healer3.hp,
      healer_3_x: healer3.x,
      healer_3_y: healer3.y,
      healer_3_aggro: healer3.aggro,

      // Starting doses for normalization
      starting_super_combat_doses: this.startingSuperCombatDoses,
      starting_sara_brew_doses: this.startingSaraBrewDoses,
      starting_super_restore_doses: this.startingSuperRestoreDoses,
    };
  }

}

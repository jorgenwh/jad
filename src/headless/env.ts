/**
 * Headless environment wrapper for RL training.
 */

import { World, Region, Player, Potion, Settings, Trainer } from 'osrs-sdk';
import { Jad } from '../jad';
import { JadRegion } from '../jad-region';

// Initialize settings from storage (uses mock localStorage)
Settings.readFromStorage();

// Action enum
export enum JadAction {
  WAIT = 0,
  PRAY_MAGE = 1,
  PRAY_RANGE = 2,
  DRINK_RESTORE = 3,
  ATTACK = 4,
}

export interface JadObservation {
  player_hp: number;
  player_prayer: number;
  active_prayer: number; // 0=none, 1=mage, 2=range
  jad_hp: number;
  jad_attack: number; // 0=none, 1=mage, 2=range
  restore_doses: number;
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
  private currentJadAttack: number = 0; // 0=none, 1=mage, 2=range
  private attackTicksRemaining: number = 0;
  private prevJadAttackDelay: number = 0; // To detect when Jad actually attacks

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
    switch (action) {
      case JadAction.WAIT:
        // Do nothing
        break;

      case JadAction.PRAY_MAGE:
        this.setPrayer('magic');
        break;

      case JadAction.PRAY_RANGE:
        this.setPrayer('range');
        break;

      case JadAction.DRINK_RESTORE:
        this.drinkRestore();
        break;

      case JadAction.ATTACK:
        this.attackJad();
        break;
    }
  }

  private setPrayer(style: 'magic' | 'range'): void {
    const prayerController = this.player.prayerController;
    if (!prayerController) return;

    // Map style to prayer name
    const prayerName = style === 'magic' ? 'Protect from Magic' : 'Protect from Range';

    // Find the prayer by name (in all prayers, not just active ones)
    const targetPrayer = prayerController.findPrayerByName(prayerName);

    if (targetPrayer && !targetPrayer.isActive) {
      // Activate the prayer - this auto-deactivates conflicting overhead prayers
      targetPrayer.activate(this.player);
    }
  }

  // TODO: find a better way to identify restore potions
  private drinkRestore(): void {
    // Find a super restore in inventory and drink it
    if (!this.player || !this.player.inventory) return;
    const inventory = this.player.inventory;

    for (const item of inventory) {
      if (item && item instanceof Potion && item.doses > 0) {
        // Check if it's a restore-type potion by checking itemName
        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes('restore')) {
          item.drink(this.player);
          break;
        }
      }
    }
  }

  private attackJad(): void {
    // Set Jad as aggro target
    this.player.setAggro(this.jad);
  }

  private updateJadAttackTracking(): void {
    // Track Jad's attack style
    // The attack style is visible when Jad attacks and persists until damage lands

    const currentDelay = this.jad.attackDelay;

    // Detect actual attack: attackDelay was low (0-1), now it's high (just reset after attacking)
    if (this.prevJadAttackDelay <= 1 && currentDelay > 1) {
      // Jad just attacked - capture the attack style NOW
      const style = this.jad.attackStyle;
      this.currentJadAttack = style === 'magic' ? 1 : 2;
      this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
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
    // Get active prayer (0=none, 1=mage, 2=range)
    let activePrayer = 0;
    const prayerController = this.player.prayerController;
    if (prayerController) {
      // Check which overhead protection prayer is active by name
      const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
      const rangePrayer = prayerController.findPrayerByName('Protect from Range');

      if (magicPrayer?.isActive) {
        activePrayer = 1;
      } else if (rangePrayer?.isActive) {
        activePrayer = 2;
      }
    }

    // Count restore doses
    let restoreDoses = 0;
    if (this.player && this.player.inventory) {
      const inventory = this.player.inventory;
      for (const item of inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('restore')) {
            restoreDoses += item.doses;
          }
        }
      }
    }

    return {
      player_hp: this.player?.currentStats?.hitpoint ?? 0,
      player_prayer: this.player?.currentStats?.prayer ?? 0,
      active_prayer: activePrayer,
      jad_hp: this.jad?.currentStats?.hitpoint ?? 0,
      jad_attack: this.currentJadAttack,
      restore_doses: restoreDoses,
    };
  }

}

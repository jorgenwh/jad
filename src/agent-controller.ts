/**
 * WebSocket client for connecting to the Python agent server.
 * Sends observations and receives actions to control the player.
 */

import { Player, Potion } from 'osrs-sdk';
import { Jad } from './jad';

// Action enum (must match Python)
enum JadAction {
  WAIT = 0,
  PRAY_MAGE = 1,      // Toggle protect from magic
  PRAY_RANGE = 2,     // Toggle protect from range
  DRINK_RESTORE = 3,
  ATTACK = 4,
  PRAY_MELEE = 5,     // Toggle protect from melee
  DRINK_SUPER_COMBAT = 6,
  TOGGLE_PIETY = 7,
  DRINK_SARA_BREW = 8,
}

interface JadObservation {
  player_hp: number;
  player_prayer: number;
  active_prayer: number; // 0=none, 1=mage, 2=range, 3=melee
  jad_hp: number;
  jad_attack: number; // 0=none, 1=mage, 2=range, 3=melee
  restore_doses: number;
  super_combat_doses: number;
  sara_brew_doses: number;
  piety_active: boolean;
  player_aggro: boolean;
}

export class AgentController {
  private ws: WebSocket | null = null;
  private player: Player;
  private jad: Jad;
  private connected = false;
  private pendingAction: number | null = null;

  // Track Jad's attack for observation
  private currentJadAttack = 0;
  private attackTicksRemaining = 0;
  private prevJadAttackDelay = 0;

  constructor(player: Player, jad: Jad) {
    this.player = player;
    this.jad = jad;
  }

  connect(url = 'ws://localhost:8765'): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to agent server at ${url}...`);
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('Connected to agent server!');
        this.connected = true;
        this.sendReset();
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'action') {
            this.pendingAction = data.action;
            console.log(`Agent action: ${data.action_name}`);
          }
        } catch (e) {
          console.error('Error parsing message:', e);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        console.log('Disconnected from agent server');
        this.connected = false;
      };
    });
  }

  private sendReset(): void {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  private sendTerminated(result: string): void {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ type: 'terminated', result }));
    }
  }

  /**
   * Called each tick to update the agent and execute actions.
   */
  tick(): void {
    if (!this.connected || !this.ws) return;

    // Check for termination
    const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
    const jadDead = this.jad.dying > 0 || this.jad.currentStats.hitpoint <= 0;

    if (playerDead) {
      this.sendTerminated('player_died');
      return;
    }
    if (jadDead) {
      this.sendTerminated('jad_killed');
      return;
    }

    // Execute pending action (from previous tick's response)
    if (this.pendingAction !== null) {
      this.executeAction(this.pendingAction);
      this.pendingAction = null;
    }

    // Update Jad attack tracking
    this.updateJadAttackTracking();

    // Send current observation
    const obs = this.getObservation();
    this.ws.send(JSON.stringify({ type: 'observation', observation: obs }));
  }

  private executeAction(action: number): void {
    switch (action) {
      case JadAction.WAIT:
        // Do nothing
        break;

      case JadAction.PRAY_MAGE:
        this.togglePrayer('Protect from Magic');
        break;

      case JadAction.PRAY_RANGE:
        this.togglePrayer('Protect from Range');
        break;

      case JadAction.DRINK_RESTORE:
        this.drinkPotion('restore');
        break;

      case JadAction.ATTACK:
        this.attackJad();
        break;

      case JadAction.PRAY_MELEE:
        this.togglePrayer('Protect from Melee');
        break;

      case JadAction.DRINK_SUPER_COMBAT:
        this.drinkPotion('super combat');
        break;

      case JadAction.TOGGLE_PIETY:
        this.togglePrayer('Piety');
        break;

      case JadAction.DRINK_SARA_BREW:
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
        targetPrayer.activate(this.player);
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
          item.drink(this.player);
          break;
        }
      }
    }
  }

  private attackJad(): void {
    this.player.setAggro(this.jad);
  }

  private updateJadAttackTracking(): void {
    const currentDelay = this.jad.attackDelay;

    // Detect actual attack
    if (this.prevJadAttackDelay <= 1 && currentDelay > 1) {
      const style = this.jad.attackStyle;
      if (style === 'magic') {
        this.currentJadAttack = 1;
        this.attackTicksRemaining = 4; // Projectile flight time
      } else if (style === 'range') {
        this.currentJadAttack = 2;
        this.attackTicksRemaining = 4; // Projectile flight time
      } else {
        // Melee attack - style is 'stab' (from canMeleeIfClose)
        // Melee is instant, no projectile delay
        // Use 2 so after immediate decrement it's 1 (visible for 1 tick)
        this.currentJadAttack = 3;
        this.attackTicksRemaining = 2;
      }
    }

    if (this.attackTicksRemaining > 0) {
      this.attackTicksRemaining--;
      if (this.attackTicksRemaining === 0) {
        this.currentJadAttack = 0;
      }
    }

    this.prevJadAttackDelay = currentDelay;
  }

  private getObservation(): JadObservation {
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

    // Count potion doses
    let restoreDoses = 0;
    let superCombatDoses = 0;
    let saraBrewDoses = 0;
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('restore')) {
            restoreDoses += item.doses;
          } else if (itemName.includes('super combat')) {
            superCombatDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            saraBrewDoses += item.doses;
          }
        }
      }
    }

    // Check if player is attacking Jad
    const playerAggro = this.player.aggro === this.jad;

    return {
      player_hp: this.player?.currentStats?.hitpoint ?? 0,
      player_prayer: this.player?.currentStats?.prayer ?? 0,
      active_prayer: activePrayer,
      jad_hp: this.jad?.currentStats?.hitpoint ?? 0,
      jad_attack: this.currentJadAttack,
      restore_doses: restoreDoses,
      super_combat_doses: superCombatDoses,
      sara_brew_doses: saraBrewDoses,
      piety_active: pietyActive,
      player_aggro: playerAggro,
    };
  }
}

/**
 * WebSocket client for connecting to the Python agent server.
 * Sends observations and receives actions to control the player.
 */

import { Player, Potion } from 'osrs-sdk';
import { Jad } from './jad';

// Action enum (must match Python)
enum JadAction {
  WAIT = 0,
  PRAY_MAGE = 1,
  PRAY_RANGE = 2,
  DRINK_RESTORE = 3,
  ATTACK = 4,
}

interface JadObservation {
  player_hp: number;
  player_prayer: number;
  active_prayer: number;
  jad_hp: number;
  jad_attack: number;
  restore_doses: number;
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

    const prayerName = style === 'magic' ? 'Protect from Magic' : 'Protect from Range';
    const targetPrayer = prayerController.findPrayerByName(prayerName);

    if (targetPrayer && !targetPrayer.isActive) {
      targetPrayer.activate(this.player);
    }
  }

  private drinkRestore(): void {
    if (!this.player || !this.player.inventory) return;
    const inventory = this.player.inventory;

    for (const item of inventory) {
      if (item && item instanceof Potion && item.doses > 0) {
        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes('restore')) {
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
      this.currentJadAttack = style === 'magic' ? 1 : 2;
      this.attackTicksRemaining = 4;
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
    let activePrayer = 0;
    const prayerController = this.player.prayerController;
    if (prayerController) {
      const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
      const rangePrayer = prayerController.findPrayerByName('Protect from Range');

      if (magicPrayer?.isActive) {
        activePrayer = 1;
      } else if (rangePrayer?.isActive) {
        activePrayer = 2;
      }
    }

    let restoreDoses = 0;
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('restore')) {
            restoreDoses += item.doses;
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
      player_aggro: playerAggro,
    };
  }
}

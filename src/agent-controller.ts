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

// Prayer names for display
const PRAYER_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const ATTACK_NAMES = ['None', 'Mage', 'Range', 'Melee'];

export class AgentController {
  private ws: WebSocket | null = null;
  private player: Player;
  private jad: Jad;
  private connected = false;

  // Track Jad's attack for observation
  private currentJadAttack = 0;
  private attackTicksRemaining = 0;
  private prevJadAttackDelay = 0;

  // Track previous state for reward calculation
  private prevPlayerHp = 0;
  private prevJadHp = 0;
  private prevJadAttack = 0;
  private cumulativeReward = 0;
  private episodeLength = 0;
  private episodeTerminated = false;

  constructor(player: Player, jad: Jad) {
    this.player = player;
    this.jad = jad;
    this.prevPlayerHp = player.currentStats?.hitpoint ?? 99;
    this.prevJadHp = jad.currentStats?.hitpoint ?? 350;
  }

  connect(url = 'ws://localhost:8765'): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to agent server at ${url}...`);
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('Connected to agent server!');
        this.connected = true;
        this.showAgentInfo(true);
        this.sendReset();
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'action') {
            // Execute action IMMEDIATELY when received (between ticks)
            // This ensures the action is applied before the next tick
            this.executeAction(data.action);

            // Log the actual processed array that was fed to the model
            if (data.processed_obs) {
              console.log(`Agent: ${data.action_name} | Processed obs: [${data.processed_obs.map((v: number) => v.toFixed(2)).join(', ')}]`);
            }

            // Update agent info display using observation echoed from server
            // This shows exactly what the agent received, catching any bugs
            this.updateActionDisplay(data.action_name, data.value);
            if (data.observation) {
              this.updateObservationDisplay(data.observation);
              this.updateRewardDisplay(data.observation);
            }
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
        this.showAgentInfo(false);
      };
    });
  }

  private sendReset(): void {
    if (this.ws && this.connected) {
      // Reset reward tracking
      this.cumulativeReward = 0;
      this.episodeLength = 0;
      this.prevPlayerHp = this.player.currentStats?.hitpoint ?? 99;
      this.prevJadHp = this.jad.currentStats?.hitpoint ?? 350;
      this.prevJadAttack = 0;
      this.episodeTerminated = false;
      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  private sendTerminated(result: string): void {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ type: 'terminated', result }));
    }
  }

  /**
   * Called each tick AFTER the world has ticked.
   * Observes the new state and sends it to the agent.
   * Actions are executed immediately in onmessage, not here.
   */
  tick(): void {
    if (!this.connected || !this.ws) return;

    // Don't update after episode has ended - freeze display at final state
    if (this.episodeTerminated) return;

    // Check for termination
    const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
    const jadDead = this.jad.dying > 0 || this.jad.currentStats.hitpoint <= 0;

    if (playerDead) {
      // Show final state with death penalty (matches rewards.py: -100)
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      this.cumulativeReward += -100;
      const rewardEl = document.getElementById('agent_reward');
      if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
      const actionEl = document.getElementById('agent_action');
      if (actionEl) actionEl.innerText = 'DEAD';

      this.sendTerminated('player_died');
      return;
    }
    if (jadDead) {
      // Show final state with kill bonus (matches rewards.py: +100 - episode_length * 0.25)
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      const killReward = 100 - this.episodeLength * 0.25;
      this.cumulativeReward += killReward;
      const rewardEl = document.getElementById('agent_reward');
      if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
      const actionEl = document.getElementById('agent_action');
      if (actionEl) actionEl.innerText = 'WIN';

      this.sendTerminated('jad_killed');
      return;
    }

    // Update Jad attack tracking (after tick, same as headless)
    this.updateJadAttackTracking();

    // Send current observation (state AFTER tick, matching headless)
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

    // Check if we can drink (cooldown check)
    const canDrink = this.player.eats?.canDrinkPotion() ?? true;
    if (!canDrink) {
      console.log(`Cannot drink ${potionType} - on cooldown (potionDelay: ${this.player.eats?.potionDelay})`);
      return;
    }

    for (const item of inventory) {
      if (item && item instanceof Potion && item.doses > 0) {
        const itemName = item.itemName?.toString().toLowerCase() || '';
        if (itemName.includes(potionType)) {
          // Use inventoryLeftClick to properly go through eating system
          // This decrements doses and queues the potion effect for next tick
          const dosesBefore = item.doses;
          item.inventoryLeftClick(this.player);
          console.log(`Drank ${potionType}: doses ${dosesBefore} -> ${item.doses}, potionDelay: ${this.player.eats?.potionDelay}`);
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

  private showAgentInfo(show: boolean): void {
    const agentInfo = document.getElementById('agent_info');
    if (agentInfo) {
      agentInfo.style.display = show ? 'block' : 'none';
    }
  }

  private updateActionDisplay(actionName: string, value: number): void {
    const actionEl = document.getElementById('agent_action');
    const valueEl = document.getElementById('agent_value');
    if (actionEl) actionEl.innerText = actionName;
    if (valueEl) valueEl.innerText = value.toFixed(2);
  }

  private updateRewardDisplay(obs: JadObservation): void {
    // Calculate step reward (matching Python rewards.py)
    let reward = 0;

    // Prayer switching feedback
    if (this.prevJadAttack !== 0) {
      if (obs.active_prayer === this.prevJadAttack) {
        reward += 1;
      } else {
        reward -= 1;
      }
    }

    // Survival reward
    reward += 0.1;

    // Damage dealt reward (damage_dealt * 0.1)
    const jadDamage = this.prevJadHp - obs.jad_hp;
    if (jadDamage > 0) {
      reward += jadDamage * 0.1;
    }

    // Damage taken penalty (damage_taken * 0.1)
    const playerDamage = this.prevPlayerHp - obs.player_hp;
    if (playerDamage > 0) {
      reward -= playerDamage * 0.1;
    }

    this.cumulativeReward += reward;
    this.episodeLength++;
    this.prevPlayerHp = obs.player_hp;
    this.prevJadHp = obs.jad_hp;
    this.prevJadAttack = obs.jad_attack;

    const rewardEl = document.getElementById('agent_reward');
    if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
  }

  private updateObservationDisplay(obs: JadObservation): void {
    const setEl = (id: string, value: string) => {
      const el = document.getElementById(id);
      if (el) el.innerText = value;
    };

    setEl('obs_hp', String(obs.player_hp));
    setEl('obs_prayer', String(obs.player_prayer));
    setEl('obs_active_prayer', PRAYER_NAMES[obs.active_prayer]);
    setEl('obs_piety', obs.piety_active ? 'Yes' : 'No');
    setEl('obs_aggro', obs.player_aggro ? 'Yes' : 'No');
    setEl('obs_jad_hp', String(obs.jad_hp));
    setEl('obs_jad_attack', ATTACK_NAMES[obs.jad_attack]);
    setEl('obs_restore', String(obs.restore_doses));
    setEl('obs_super_combat', String(obs.super_combat_doses));
    setEl('obs_sara_brew', String(obs.sara_brew_doses));
  }
}

/**
 * WebSocket client for connecting to the Python agent server.
 * Sends observations and receives actions to control the player.
 */

import { Player, Potion, UnitTypes } from 'osrs-sdk';
import { Jad } from './jad';
import { JadRegion, HealerAggro } from './jad-region';

// Action enum (must match Python)
enum JadAction {
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
enum PlayerAggro {
  NONE = 0,
  JAD = 1,
  HEALER_1 = 2,
  HEALER_2 = 3,
  HEALER_3 = 4,
}

interface JadObservation {
  // Player state
  player_hp: number;
  player_prayer: number;
  player_attack: number;
  player_strength: number;
  player_defence: number;
  player_x: number;
  player_y: number;
  player_aggro: number;

  // Prayer state
  active_prayer: number;
  piety_active: boolean;

  // Inventory
  super_combat_doses: number;
  sara_brew_doses: number;
  super_restore_doses: number;

  // Jad state
  jad_hp: number;
  jad_attack: number;
  jad_x: number;
  jad_y: number;

  // Healer state
  healers_spawned: boolean;
  healer_1_hp: number;
  healer_1_x: number;
  healer_1_y: number;
  healer_1_aggro: number;
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

// Names for display
const PRAYER_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const ATTACK_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const AGGRO_NAMES = ['None', 'Jad', 'Healer 1', 'Healer 2', 'Healer 3'];
const HEALER_AGGRO_NAMES = ['Not Present', 'Jad', 'Player'];

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
  private prevHealer1Aggro = 0;
  private prevHealer2Aggro = 0;
  private prevHealer3Aggro = 0;
  private cumulativeReward = 0;
  private episodeLength = 0;
  private episodeTerminated = false;

  // Track starting potion doses for normalization
  private startingSuperCombatDoses = 0;
  private startingSaraBrewDoses = 0;
  private startingSuperRestoreDoses = 0;

  constructor(player: Player, jad: Jad) {
    this.player = player;
    this.jad = jad;
    this.prevPlayerHp = player.currentStats?.hitpoint ?? 99;
    this.prevJadHp = jad.currentStats?.hitpoint ?? 350;
    this.captureStartingDoses();
  }

  private captureStartingDoses(): void {
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('super combat')) {
            this.startingSuperCombatDoses += item.doses;
          } else if (itemName.includes('saradomin brew')) {
            this.startingSaraBrewDoses += item.doses;
          } else if (itemName.includes('restore')) {
            this.startingSuperRestoreDoses += item.doses;
          }
        }
      }
    }
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
      this.prevHealer1Aggro = 0;
      this.prevHealer2Aggro = 0;
      this.prevHealer3Aggro = 0;
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
      // Show final state with death penalty (matches rewards.py: -50)
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      this.cumulativeReward += -50;
      const rewardEl = document.getElementById('agent_reward');
      if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
      const stepsEl = document.getElementById('agent_steps');
      if (stepsEl) stepsEl.innerText = String(this.episodeLength);
      const actionEl = document.getElementById('agent_action');
      if (actionEl) actionEl.innerText = 'DEAD';

      this.sendTerminated('player_died');
      return;
    }
    if (jadDead) {
      // Show final state with kill bonus (matches rewards.py: +100 - episode_length * 0.1)
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      const killReward = 100 - this.episodeLength * 0.1;
      this.cumulativeReward += killReward;
      const rewardEl = document.getElementById('agent_reward');
      if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
      const stepsEl = document.getElementById('agent_steps');
      if (stepsEl) stepsEl.innerText = String(this.episodeLength);
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
    const jadRegion = this.player.region as JadRegion;

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
    const jadRegion = this.player.region as JadRegion;

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
          playerAggro = PlayerAggro.HEALER_1 + i;
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

    // Healer aggro constants (must match HealerAggro enum)
    const HEALER_AGGRO_JAD = 1;
    const HEALER_AGGRO_PLAYER = 2;

    // Prayer switching feedback
    if (this.prevJadAttack !== 0) {
      if (obs.active_prayer === this.prevJadAttack) {
        reward += 1;
      } else {
        reward -= 1;
      }
    }

    // Penalty for not being in combat (encourages attacking)
    // player_aggro is now a number: 0=none, 1=jad, 2-4=healer
    if (obs.player_aggro === PlayerAggro.NONE) {
      reward -= 0.5;
    }

    // Damage dealt reward - only for Jad damage (damage_dealt * 0.2)
    const jadDamage = this.prevJadHp - obs.jad_hp;
    if (jadDamage > 0) {
      reward += jadDamage * 0.2;
    }

    // Jad healing penalty - punishes letting healers heal Jad
    const jadHealed = obs.jad_hp - this.prevJadHp;
    if (jadHealed > 0) {
      reward -= jadHealed * 0.3;
    }

    // Damage taken penalty (damage_taken * 0.1)
    const playerDamage = this.prevPlayerHp - obs.player_hp;
    if (playerDamage > 0) {
      reward -= playerDamage * 0.1;
    }

    // Healer tagging reward (+5 per healer when aggro changes from JAD to PLAYER)
    if (this.prevHealer1Aggro === HEALER_AGGRO_JAD && obs.healer_1_aggro === HEALER_AGGRO_PLAYER) {
      reward += 5;
    }
    if (this.prevHealer2Aggro === HEALER_AGGRO_JAD && obs.healer_2_aggro === HEALER_AGGRO_PLAYER) {
      reward += 5;
    }
    if (this.prevHealer3Aggro === HEALER_AGGRO_JAD && obs.healer_3_aggro === HEALER_AGGRO_PLAYER) {
      reward += 5;
    }

    this.cumulativeReward += reward;
    this.episodeLength++;
    this.prevPlayerHp = obs.player_hp;
    this.prevJadHp = obs.jad_hp;
    this.prevJadAttack = obs.jad_attack;
    this.prevHealer1Aggro = obs.healer_1_aggro;
    this.prevHealer2Aggro = obs.healer_2_aggro;
    this.prevHealer3Aggro = obs.healer_3_aggro;

    const rewardEl = document.getElementById('agent_reward');
    if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);
  }

  private updateObservationDisplay(obs: JadObservation): void {
    const setEl = (id: string, value: string) => {
      const el = document.getElementById(id);
      if (el) el.innerText = value;
    };

    // Player state
    setEl('obs_hp', String(obs.player_hp));
    setEl('obs_prayer', String(obs.player_prayer));
    setEl('obs_attack', String(obs.player_attack));
    setEl('obs_strength', String(obs.player_strength));
    setEl('obs_defence', String(obs.player_defence));
    setEl('obs_pos', `(${obs.player_x}, ${obs.player_y})`);
    setEl('obs_aggro', AGGRO_NAMES[obs.player_aggro] || 'Unknown');

    // Prayer state
    setEl('obs_active_prayer', PRAYER_NAMES[obs.active_prayer]);
    setEl('obs_piety', obs.piety_active ? 'Yes' : 'No');

    // Inventory
    setEl('obs_super_combat', String(obs.super_combat_doses));
    setEl('obs_sara_brew', String(obs.sara_brew_doses));
    setEl('obs_restore', String(obs.super_restore_doses));

    // Jad state
    setEl('obs_jad_hp', String(obs.jad_hp));
    setEl('obs_jad_attack', ATTACK_NAMES[obs.jad_attack]);
    setEl('obs_jad_pos', `(${obs.jad_x}, ${obs.jad_y})`);

    // Healer state
    setEl('obs_healers_spawned', obs.healers_spawned ? 'Yes' : 'No');

    // Healer 1
    setEl('obs_healer_1_hp', String(obs.healer_1_hp));
    setEl('obs_healer_1_pos', `(${obs.healer_1_x}, ${obs.healer_1_y})`);
    setEl('obs_healer_1_aggro', HEALER_AGGRO_NAMES[obs.healer_1_aggro]);

    // Healer 2
    setEl('obs_healer_2_hp', String(obs.healer_2_hp));
    setEl('obs_healer_2_pos', `(${obs.healer_2_x}, ${obs.healer_2_y})`);
    setEl('obs_healer_2_aggro', HEALER_AGGRO_NAMES[obs.healer_2_aggro]);

    // Healer 3
    setEl('obs_healer_3_hp', String(obs.healer_3_hp));
    setEl('obs_healer_3_pos', `(${obs.healer_3_x}, ${obs.healer_3_y})`);
    setEl('obs_healer_3_aggro', HEALER_AGGRO_NAMES[obs.healer_3_aggro]);
  }
}

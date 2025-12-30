/**
 * WebSocket client for connecting to the Python agent server.
 * Sends observations and receives actions to control the player.
 * Supports multi-Jad environments (1-6 Jads).
 */

import { Player, Potion } from 'osrs-sdk';
import { JadRegion, HealerAggro } from './jad-region';
import { JadConfig } from './config';

// Single Jad state in observation
interface JadState {
  hp: number;
  attack: number;  // 0=none, 1=mage, 2=range, 3=melee
  x: number;
  y: number;
  alive: boolean;
}

// Single healer state in observation
interface HealerState {
  hp: number;
  x: number;
  y: number;
  aggro: number;  // HealerAggro enum: 0=not_present, 1=jad, 2=player
}

interface JadObservation {
  // Config
  jad_count: number;
  healers_per_jad: number;

  // Player state
  player_hp: number;
  player_prayer: number;
  player_ranged: number;
  player_defence: number;
  player_x: number;
  player_y: number;
  player_aggro: number;  // 0=none, 1..N=jad_N, N+1..=healer

  // Prayer state
  active_prayer: number;  // 0=none, 1=mage, 2=range, 3=melee
  rigour_active: boolean;

  // Inventory
  bastion_doses: number;
  sara_brew_doses: number;
  super_restore_doses: number;

  // Dynamic Jad state array
  jads: JadState[];

  // Dynamic healer state array (flattened)
  healers: HealerState[];

  // Whether any healers have spawned
  healers_spawned: boolean;

  // Starting doses for normalization
  starting_bastion_doses: number;
  starting_sara_brew_doses: number;
  starting_super_restore_doses: number;
}

// Names for display
const PRAYER_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const ATTACK_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const HEALER_AGGRO_NAMES = ['Not Present', 'Jad', 'Player'];

// Healer aggro constant for initialization
const HEALER_AGGRO_NOT_PRESENT = 0;

export class AgentController {
  private ws: WebSocket | null = null;
  private player: Player;
  private jadRegion: JadRegion;
  private config: JadConfig;
  private connected = false;

  // Track attack state per Jad
  private jadAttackStates: { attack: number; ticksRemaining: number; prevDelay: number }[] = [];

  // Track previous state for reward calculation
  private prevPlayerHp = 0;
  private prevJadHps: number[] = [];
  private prevJadAttacks: number[] = [];
  private prevHealerAggros: number[] = [];
  private cumulativeReward = 0;
  private episodeLength = 0;
  private episodeTerminated = false;

  // Track starting potion doses for normalization
  private startingBastionDoses = 0;
  private startingSaraBrewDoses = 0;
  private startingSuperRestoreDoses = 0;

  constructor(player: Player, jadRegion: JadRegion) {
    this.player = player;
    this.jadRegion = jadRegion;
    this.config = { jadCount: jadRegion.jadCount, healersPerJad: jadRegion.healersPerJad };

    // Initialize attack tracking per Jad
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = jadRegion.getJad(i);
      this.jadAttackStates.push({
        attack: 0,
        ticksRemaining: 0,
        prevDelay: jad?.attackDelay ?? 0,
      });
      this.prevJadHps.push(jad?.currentStats?.hitpoint ?? 350);
      this.prevJadAttacks.push(0);
    }

    // Initialize healer aggro tracking
    const totalHealers = this.config.jadCount * this.config.healersPerJad;
    for (let i = 0; i < totalHealers; i++) {
      this.prevHealerAggros.push(HEALER_AGGRO_NOT_PRESENT);
    }

    this.prevPlayerHp = player.currentStats?.hitpoint ?? 99;
    this.captureStartingDoses();
  }

  private captureStartingDoses(): void {
    if (this.player && this.player.inventory) {
      for (const item of this.player.inventory) {
        if (item && item instanceof Potion && item.doses > 0) {
          const itemName = item.itemName?.toString().toLowerCase() || '';
          if (itemName.includes('bastion')) {
            this.startingBastionDoses += item.doses;
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
            this.executeAction(data.action);

            if (data.processed_obs) {
              console.log(`Agent: ${data.action_name} | Processed obs: [${data.processed_obs.map((v: number) => v.toFixed(2)).join(', ')}]`);
            }

            this.updateActionDisplay(data.action_name, data.value);
            if (data.observation) {
              this.updateObservationDisplay(data.observation);
            }
            // Use server-computed reward (single source of truth)
            if (data.cumulative_reward !== undefined) {
              this.cumulativeReward = data.cumulative_reward;
              this.episodeLength = data.episode_length;
              this.displayReward();
            }
          } else if (data.type === 'terminated_ack') {
            // Server sent final reward after episode termination
            this.cumulativeReward = data.cumulative_reward;
            this.episodeLength = data.episode_length;
            this.displayReward();
            console.log(`Episode ${data.result}: terminal_reward=${data.terminal_reward.toFixed(1)}, total=${data.cumulative_reward.toFixed(1)}`);
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
      this.cumulativeReward = 0;
      this.episodeLength = 0;
      this.prevPlayerHp = this.player.currentStats?.hitpoint ?? 99;
      this.episodeTerminated = false;

      // Reset per-Jad tracking
      for (let i = 0; i < this.config.jadCount; i++) {
        const jad = this.jadRegion.getJad(i);
        this.prevJadHps[i] = jad?.currentStats?.hitpoint ?? 350;
        this.prevJadAttacks[i] = 0;
        this.jadAttackStates[i] = {
          attack: 0,
          ticksRemaining: 0,
          prevDelay: jad?.attackDelay ?? 0,
        };
      }

      // Reset healer aggro tracking
      for (let i = 0; i < this.prevHealerAggros.length; i++) {
        this.prevHealerAggros[i] = HEALER_AGGRO_NOT_PRESENT;
      }

      this.ws.send(JSON.stringify({ type: 'reset' }));
    }
  }

  private sendTerminated(result: string, observation: JadObservation): void {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify({ type: 'terminated', result, observation }));
    }
  }

  tick(): void {
    if (!this.connected || !this.ws) return;
    if (this.episodeTerminated) return;

    // Check for termination
    const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;

    // Check if ALL Jads are dead
    let allJadsDead = true;
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = this.jadRegion.getJad(i);
      if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
        allJadsDead = false;
        break;
      }
    }

    if (playerDead) {
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      // Server computes terminal reward - just update display
      const actionEl = document.getElementById('agent_action');
      if (actionEl) actionEl.innerText = 'DEAD';
      this.sendTerminated('player_died', obs);
      return;
    }

    if (allJadsDead) {
      this.episodeTerminated = true;
      const obs = this.getObservation();
      this.updateObservationDisplay(obs);
      // Server computes terminal reward - just update display
      const actionEl = document.getElementById('agent_action');
      if (actionEl) actionEl.innerText = 'WIN';
      this.sendTerminated('jad_killed', obs);
      return;
    }

    // Update Jad attack tracking for all Jads
    this.updateJadAttackTracking();

    const obs = this.getObservation();
    this.ws.send(JSON.stringify({ type: 'observation', observation: obs }));
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
      const jad = this.jadRegion.getJad(jadIndex);
      if (jad) {
        console.log(`[AGGRO] Targeting Jad ${jadIndex} (HP: ${jad.currentStats.hitpoint})`);
        this.player.setAggro(jad);
      } else {
        // Debug: check why jad is null
        const rawJad = (this.jadRegion as any)._jads[jadIndex];
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
      const healer = this.jadRegion.getHealer(jadIndex, healerIndex);
      if (healer) {
        console.log(`[AGGRO] Targeting Healer [${jadIndex}][${healerIndex}] (HP: ${healer.currentStats.hitpoint})`);
        this.player.setAggro(healer);
      } else {
        // Debug: check why healer is null
        const healerTuple = (this.jadRegion as any)._healers.get(jadIndex);
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
      case 0:
        this.togglePrayer('Protect from Melee');
        break;
      case 1:
        this.togglePrayer('Protect from Range');
        break;
      case 2:
        this.togglePrayer('Protect from Magic');
        break;
      case 3:
        this.togglePrayer('Rigour');
        break;
      case 4:
        this.drinkPotion('bastion');
        break;
      case 5:
        this.drinkPotion('restore');
        break;
      case 6:
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

    const canDrink = this.player.eats?.canDrinkPotion() ?? true;
    if (!canDrink) {
      console.log(`Cannot drink ${potionType} - on cooldown`);
      return;
    }

    for (const item of this.player.inventory) {
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
    for (let i = 0; i < this.config.jadCount; i++) {
      const jad = this.jadRegion.getJad(i);
      if (!jad) continue;

      const state = this.jadAttackStates[i];
      const currentDelay = jad.attackDelay;

      if (state.prevDelay <= 1 && currentDelay > 1) {
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

      if (state.ticksRemaining > 0) {
        state.ticksRemaining--;
        if (state.ticksRemaining === 0) {
          state.attack = 0;
        }
      }

      state.prevDelay = currentDelay;
    }
  }

  private getObservation(): JadObservation {
    // Get active prayer
    let activePrayer = 0;
    let rigourActive = false;
    const prayerController = this.player.prayerController;
    if (prayerController) {
      const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
      const rangePrayer = prayerController.findPrayerByName('Protect from Range');
      const meleePrayer = prayerController.findPrayerByName('Protect from Melee');
      const rigourPrayer = prayerController.findPrayerByName('Rigour');

      if (magicPrayer?.isActive) activePrayer = 1;
      else if (rangePrayer?.isActive) activePrayer = 2;
      else if (meleePrayer?.isActive) activePrayer = 3;

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
        const jad = this.jadRegion.getJad(i);
        if (jad && playerAggroTarget === jad) {
          playerAggro = i + 1;
          break;
        }
      }

      // If not a Jad, check if it's a healer
      if (playerAggro === 0) {
        const numJads = this.config.jadCount;
        const healersPerJad = this.config.healersPerJad;
        for (let jadIdx = 0; jadIdx < numJads; jadIdx++) {
          for (let healerIdx = 0; healerIdx < healersPerJad; healerIdx++) {
            const healer = this.jadRegion.getHealer(jadIdx, healerIdx);
            if (healer && playerAggroTarget === healer) {
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
      const jad = this.jadRegion.getJad(i);
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
        const healer = this.jadRegion.getHealer(jadIdx, healerIdx);
        if (healer) {
          healersSpawned = true;
          healers.push({
            hp: healer.currentStats?.hitpoint ?? 0,
            x: healer.location.x,
            y: healer.location.y,
            aggro: this.jadRegion.getHealerAggro(jadIdx, healerIdx),
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
      jad_count: this.config.jadCount,
      healers_per_jad: this.config.healersPerJad,

      player_hp: this.player?.currentStats?.hitpoint ?? 0,
      player_prayer: this.player?.currentStats?.prayer ?? 0,
      player_ranged: this.player?.currentStats?.range ?? 99,
      player_defence: this.player?.currentStats?.defence ?? 99,
      player_x: this.player?.location?.x ?? 0,
      player_y: this.player?.location?.y ?? 0,
      player_aggro: playerAggro,

      active_prayer: activePrayer,
      rigour_active: rigourActive,

      bastion_doses: bastionDoses,
      sara_brew_doses: saraBrewDoses,
      super_restore_doses: superRestoreDoses,

      jads: jads,
      healers: healers,
      healers_spawned: healersSpawned,

      starting_bastion_doses: this.startingBastionDoses,
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

  private displayReward(): void {
    // Display reward computed by Python server (single source of truth)
    const rewardEl = document.getElementById('agent_reward');
    if (rewardEl) rewardEl.innerText = this.cumulativeReward.toFixed(1);

    const stepsEl = document.getElementById('agent_steps');
    if (stepsEl) stepsEl.innerText = String(this.episodeLength);
  }

  private updateObservationDisplay(obs: JadObservation): void {
    const setEl = (id: string, value: string) => {
      const el = document.getElementById(id);
      if (el) el.innerText = value;
    };

    // Config
    setEl('obs_jad_count', String(obs.jad_count));
    setEl('obs_healers_per_jad', String(obs.healers_per_jad));

    // Player state
    setEl('obs_hp', String(obs.player_hp));
    setEl('obs_prayer', String(obs.player_prayer));
    setEl('obs_ranged', String(obs.player_ranged));
    setEl('obs_defence', String(obs.player_defence));
    setEl('obs_pos', `(${obs.player_x}, ${obs.player_y})`);

    // Player aggro (decode dynamic format)
    const numJads = obs.jad_count;
    const healersPerJad = obs.healers_per_jad;
    let aggroName = 'None';
    if (obs.player_aggro >= 1 && obs.player_aggro <= numJads) {
      aggroName = `Jad ${obs.player_aggro}`;
    } else if (obs.player_aggro > numJads) {
      const healerIdx = obs.player_aggro - numJads - 1;
      const jadIdx = Math.floor(healerIdx / healersPerJad);
      const hIdx = healerIdx % healersPerJad;
      aggroName = `H${jadIdx + 1}.${hIdx + 1}`;
    }
    setEl('obs_aggro', aggroName);

    // Prayer state
    setEl('obs_active_prayer', PRAYER_NAMES[obs.active_prayer]);
    setEl('obs_rigour', obs.rigour_active ? '1' : '0');

    // Inventory
    setEl('obs_bastion', String(obs.bastion_doses));
    setEl('obs_sara_brew', String(obs.sara_brew_doses));
    setEl('obs_restore', String(obs.super_restore_doses));

    // Jads - build dynamic display
    const jadsContainer = document.getElementById('obs_jads_container');
    if (jadsContainer) {
      let html = '';
      for (let i = 0; i < obs.jads.length; i++) {
        const jad = obs.jads[i];
        const attackName = ATTACK_NAMES[jad.attack] || 'Unknown';
        const status = jad.alive ? '' : ' (dead)';
        html += `<div class="obs-jad">`;
        html += `<strong>Jad ${i + 1}${status}:</strong> `;
        html += `HP: ${jad.hp} | Atk: ${attackName} | Pos: (${jad.x}, ${jad.y})`;
        html += `</div>`;
      }
      jadsContainer.innerHTML = html;
    }

    // Healers - build dynamic display
    setEl('obs_healers_spawned', obs.healers_spawned ? '1' : '0');
    const healersContainer = document.getElementById('obs_healers_container');
    if (healersContainer) {
      let html = '';
      for (let i = 0; i < obs.healers.length; i++) {
        const healer = obs.healers[i];
        const jadIdx = Math.floor(i / obs.healers_per_jad);
        const hIdx = i % obs.healers_per_jad;
        const aggroName = HEALER_AGGRO_NAMES[healer.aggro] || 'Unknown';
        html += `<div class="obs-healer">`;
        html += `<strong>H${jadIdx + 1}.${hIdx + 1}:</strong> `;
        html += `HP: ${healer.hp} | Aggro: ${aggroName} | Pos: (${healer.x}, ${healer.y})`;
        html += `</div>`;
      }
      healersContainer.innerHTML = html;
    }
  }
}

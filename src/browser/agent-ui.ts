/**
 * UI service for displaying agent state in the browser.
 * Handles all DOM updates for observation display, action display, and rewards.
 */

import { JadObservation } from '../core';

const PRAYER_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const ATTACK_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const HEALER_AGGRO_NAMES = ['Not Present', 'Jad', 'Player'];

export class AgentUI {
    showAgentInfo(show: boolean): void {
        const agentInfo = document.getElementById('agent_info');
        if (agentInfo) {
            agentInfo.style.display = show ? 'block' : 'none';
        }
    }

    updateAction(actionName: string, value: number): void {
        this.setElement('agent_action', actionName);
        this.setElement('agent_value', value.toFixed(2));
    }

    updateActionStatus(status: string): void {
        this.setElement('agent_action', status);
    }

    updateReward(cumulativeReward: number, episodeLength: number): void {
        this.setElement('agent_reward', cumulativeReward.toFixed(1));
        this.setElement('agent_steps', String(episodeLength));
    }

    updateObservation(obs: JadObservation): void {
        // Config
        this.setElement('obs_jad_count', String(obs.jad_count));
        this.setElement('obs_healers_per_jad', String(obs.healers_per_jad));

        // Player state
        this.setElement('obs_hp', String(obs.player_hp));
        this.setElement('obs_prayer', String(obs.player_prayer));
        this.setElement('obs_ranged', String(obs.player_ranged));
        this.setElement('obs_defence', String(obs.player_defence));
        this.setElement('obs_pos', `(${obs.player_x}, ${obs.player_y})`);

        // Player aggro
        this.setElement('obs_aggro', this.decodeAggroName(obs));

        // Prayer state
        this.setElement('obs_active_prayer', PRAYER_NAMES[obs.active_prayer]);
        this.setElement('obs_rigour', obs.rigour_active ? '1' : '0');

        // Inventory
        this.setElement('obs_bastion', String(obs.bastion_doses));
        this.setElement('obs_sara_brew', String(obs.sara_brew_doses));
        this.setElement('obs_restore', String(obs.super_restore_doses));

        // Jads
        this.updateJadsDisplay(obs);

        // Healers
        this.setElement('obs_healers_spawned', obs.healers_spawned ? '1' : '0');
        this.updateHealersDisplay(obs);
    }

    private decodeAggroName(obs: JadObservation): string {
        const numJads = obs.jad_count;
        const healersPerJad = obs.healers_per_jad;

        if (obs.player_aggro >= 1 && obs.player_aggro <= numJads) {
            return `Jad ${obs.player_aggro}`;
        } else if (obs.player_aggro > numJads) {
            const healerIdx = obs.player_aggro - numJads - 1;
            const jadIdx = Math.floor(healerIdx / healersPerJad);
            const hIdx = healerIdx % healersPerJad;
            return `H${jadIdx + 1}.${hIdx + 1}`;
        }
        return 'None';
    }

    private updateJadsDisplay(obs: JadObservation): void {
        const container = document.getElementById('obs_jads_container');
        if (!container) return;

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
        container.innerHTML = html;
    }

    private updateHealersDisplay(obs: JadObservation): void {
        const container = document.getElementById('obs_healers_container');
        if (!container) return;

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
        container.innerHTML = html;
    }

    private setElement(id: string, value: string): void {
        const el = document.getElementById(id);
        if (el) el.innerText = value;
    }
}

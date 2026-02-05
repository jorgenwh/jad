import { Observation, JadConfig } from '../core';

const PRAYER_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const ATTACK_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const HEALER_TARGET_NAMES = ['Not Present', 'Jad', 'Player'];

const PROTECTION_NAMES = ['None', 'Mage', 'Range', 'Melee'];
const OFFENSIVE_NAMES = ['None', 'Rigour'];
const POTION_NAMES = ['None', 'Bastion', 'Restore', 'Brew'];

export class AgentUI {
    showAgentInfo(show: boolean): void {
        const agentInfo = document.getElementById('agent_info');
        if (agentInfo) {
            agentInfo.style.display = show ? 'block' : 'none';
        }
    }

    updateAction(action: number[], value: number, config: JadConfig): void {
        // Protection prayer (head 0)
        this.setElement('agent_protection', PROTECTION_NAMES[action[0]]);

        // Offensive prayer (head 1)
        this.setElement('agent_offensive', OFFENSIVE_NAMES[action[1]]);

        // Potion (head 2)
        this.setElement('agent_potion', POTION_NAMES[action[2]]);

        // Target (head 3)
        const target = action[3];
        let targetName = 'None';
        if (target > 0) {
            const numJads = config.jadCount;
            if (target <= numJads) {
                targetName = `Jad ${target}`;
            } else {
                const healerIdx = target - numJads - 1;
                const jadIdx = Math.floor(healerIdx / config.healersPerJad);
                const hIdx = healerIdx % config.healersPerJad;
                targetName = `H${jadIdx + 1}.${hIdx + 1}`;
            }
        }
        this.setElement('agent_target', targetName);

        this.setElement('agent_value', value.toFixed(2));
    }

    updateActionStatus(status: string): void {
        this.setElement('agent_status', `[${status}]`);
    }

    updateReward(cumulativeReward: number, episodeLength: number): void {
        this.setElement('agent_reward', cumulativeReward.toFixed(1));
        this.setElement('agent_steps', String(episodeLength));
    }

    updateObservation(obs: Observation): void {
        // Config (derived from arrays)
        const jadCount = obs.jads.length;
        const healersPerJad = jadCount > 0 ? obs.healers.length / jadCount : 0;
        this.setElement('obs_jad_count', String(jadCount));
        this.setElement('obs_healers_per_jad', String(healersPerJad));

        // Player state
        this.setElement('obs_hp', String(obs.player_hp));
        this.setElement('obs_prayer', String(obs.player_prayer));
        this.setElement('obs_ranged', String(obs.player_ranged));
        this.setElement('obs_defence', String(obs.player_defence));
        this.setElement('obs_pos', `(${obs.player_location_x}, ${obs.player_location_y})`);

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

    private decodeAggroName(obs: Observation): string {
        const numJads = obs.jads.length;
        const healersPerJad = numJads > 0 ? obs.healers.length / numJads : 0;

        if (obs.player_target >= 1 && obs.player_target <= numJads) {
            return `Jad ${obs.player_target}`;
        } else if (obs.player_target > numJads) {
            const healerIdx = obs.player_target - numJads - 1;
            const jadIdx = Math.floor(healerIdx / healersPerJad);
            const hIdx = healerIdx % healersPerJad;
            return `H${jadIdx + 1}.${hIdx + 1}`;
        }
        return 'None';
    }

    private updateJadsDisplay(obs: Observation): void {
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

    private updateHealersDisplay(obs: Observation): void {
        const container = document.getElementById('obs_healers_container');
        if (!container) return;

        let html = '';
        const healersPerJad = obs.jads.length > 0 ? obs.healers.length / obs.jads.length : 0;
        for (let i = 0; i < obs.healers.length; i++) {
            const healer = obs.healers[i];
            const jadIdx = Math.floor(i / healersPerJad);
            const hIdx = i % healersPerJad;
            const targetName = HEALER_TARGET_NAMES[healer.target] || 'Unknown';
            html += `<div class="obs-healer">`;
            html += `<strong>H${jadIdx + 1}.${hIdx + 1}:</strong> `;
            html += `HP: ${healer.hp} | Target: ${targetName} | Pos: (${healer.x}, ${healer.y})`;
            html += `</div>`;
        }
        container.innerHTML = html;
    }

    private setElement(id: string, value: string): void {
        const el = document.getElementById(id);
        if (el) el.innerText = value;
    }
}

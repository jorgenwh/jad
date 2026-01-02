import { Player } from 'osrs-sdk';
import { JadRegion, JadConfig, executeAction, countPotionDoses, buildObservation } from '../core';
import { AgentWebSocket } from './agent-websocket';
import { AgentUI } from './agent-ui';

type TerminationResult = 'player_died' | 'jad_killed' | null;

export class AgentController {
    private ws: AgentWebSocket;
    private ui: AgentUI;

    private player: Player;
    private jadRegion: JadRegion;
    private config: JadConfig;

    private startingDoses = { bastion: 0, saraBrew: 0, superRestore: 0 };
    private terminated = false;
    private terminationResult: TerminationResult = null;

    constructor(player: Player, jadRegion: JadRegion, config: JadConfig) {
        this.player = player;
        this.jadRegion = jadRegion;
        this.config = config;

        this.ws = new AgentWebSocket();
        this.ui = new AgentUI();

        this.captureStartingDoses();
        this.setupWebSocketHandlers();
    }

    private setupWebSocketHandlers(): void {
        this.ws.onConnectionChange = (connected) => {
            this.ui.showAgentInfo(connected);
        };

        this.ws.onAction = (action, actionName, value, observation, cumulativeReward, episodeLength, processedObs) => {
            executeAction(action, this.player, this.jadRegion, this.config);

            if (processedObs) {
                console.log(`Agent: ${actionName} | Processed obs: [${processedObs.map((v: number) => v.toFixed(2)).join(', ')}]`);
            }

            this.ui.updateAction(actionName, value);

            if (observation) {
                this.ui.updateObservation(observation);
            }

            if (cumulativeReward !== undefined && episodeLength !== undefined) {
                this.cumulativeReward = cumulativeReward;
                this.episodeLength = episodeLength;
                this.ui.updateReward(cumulativeReward, episodeLength);
            }
        };

        this.ws.onTerminatedAck = (cumulativeReward, episodeLength, terminalReward, result) => {
            this.cumulativeReward = cumulativeReward;
            this.episodeLength = episodeLength;
            this.ui.updateReward(cumulativeReward, episodeLength);
            console.log(`Episode ${result}: terminal_reward=${terminalReward.toFixed(1)}, total=${cumulativeReward.toFixed(1)}`);
        };
    }

    connect(): Promise<void> {
        return this.ws.connect().then(() => {
            this.resetEpisode();
            this.ws.sendReset();
        });
    }

    tick(): void {
        // Skip if not connected to ws server
        if (!this.ws.connected) {
            return;
        }
        // Skip if episode already terminated
        if (this.terminated) {
            return;
        }

        // Check for termination
        const result = this.checkTermination();
        if (result) {
            const obs = buildObservation(
                this.player,
                this.jadRegion,
                this.config,
                this.startingDoses
            );
            this.ui.updateObservation(obs);
            this.ui.updateActionStatus(result === 'player_died' ? 'DEAD' : 'WIN');
            this.ws.sendTerminated(result, obs);
            return;
        }

        // Send observation
        const obs = buildObservation(
            this.player,
            this.jadRegion,
            this.config,
            this.startingDoses
        );
        this.ws.sendObservation(obs);
    }

    private captureStartingDoses(): void {
        const doses = countPotionDoses(this.player);
        this.startingDoses = {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }

    private resetEpisode(): void {
        this.cumulativeReward = 0;
        this.episodeLength = 0;
        this.terminated = false;
        this.terminationResult = null;
        this.captureStartingDoses();
    }

    private checkTermination(): TerminationResult {
        if (this.terminated) {
            return this.terminationResult;
        }

        // Check player death
        const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
        if (playerDead) {
            this.terminated = true;
            this.terminationResult = 'player_died';
            return this.terminationResult;
        }

        // Check if all Jads are dead
        let allJadsDead = true;
        for (let i = 0; i < this.config.jadCount; i++) {
            const jad = this.jadRegion.getJad(i);
            if (jad && jad.currentStats.hitpoint > 0 && jad.dying === -1) {
                allJadsDead = false;
                break;
            }
        }

        if (allJadsDead) {
            this.terminated = true;
            this.terminationResult = 'jad_killed';
            return this.terminationResult;
        }

        return null;
    }
}

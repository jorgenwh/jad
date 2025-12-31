/**
 * Agent controller that orchestrates the browser-side RL agent.
 * Coordinates between WebSocket communication, episode state, and UI display.
 */

import { Player } from 'osrs-sdk';
import { JadRegion, JadConfig, EpisodeState, executeAction } from '../core';
import { AgentWebSocket } from './agent-websocket';
import { AgentUI } from './agent-ui';

export class AgentController {
    private ws: AgentWebSocket;
    private ui: AgentUI;
    private episode: EpisodeState;

    private player: Player;
    private jadRegion: JadRegion;
    private config: JadConfig;

    constructor(player: Player, jadRegion: JadRegion) {
        this.player = player;
        this.jadRegion = jadRegion;
        this.config = {
            jadCount: jadRegion.jadCount,
            healersPerJad: jadRegion.healersPerJad,
        };

        this.ws = new AgentWebSocket();
        this.ui = new AgentUI();
        this.episode = new EpisodeState(player, jadRegion, this.config);

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
                this.episode.cumulativeReward = cumulativeReward;
                this.episode.episodeLength = episodeLength;
                this.ui.updateReward(cumulativeReward, episodeLength);
            }
        };

        this.ws.onTerminatedAck = (cumulativeReward, episodeLength, terminalReward, result) => {
            this.episode.cumulativeReward = cumulativeReward;
            this.episode.episodeLength = episodeLength;
            this.ui.updateReward(cumulativeReward, episodeLength);
            console.log(`Episode ${result}: terminal_reward=${terminalReward.toFixed(1)}, total=${cumulativeReward.toFixed(1)}`);
        };
    }

    connect(): Promise<void> {
        return this.ws.connect().then(() => {
            this.episode.reset();
            this.ws.sendReset();
        });
    }

    tick(): void {
        // Skip if not connected to ws server
        if (!this.ws.connected) {
            return;
        }
        // Skip if episode already terminated
        if (this.episode.terminated) {
            return;
        }

        // Check for termination
        const result = this.episode.checkTermination();
        if (result) {
            const obs = this.episode.getObservation();
            this.ui.updateObservation(obs);
            this.ui.updateActionStatus(result === 'player_died' ? 'DEAD' : 'WIN');
            this.ws.sendTerminated(result, obs);
            return;
        }

        // Update attack tracking and send observation
        this.episode.updateAttackTracking();
        const obs = this.episode.getObservation();
        this.ws.sendObservation(obs);
    }
}

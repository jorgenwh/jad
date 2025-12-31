/**
 * WebSocket service for communicating with the Python RL agent server.
 * Handles connection management and message protocol.
 */

import { JadObservation } from '../core';

export type ActionHandler = (action: number, actionName: string, value: number, observation?: JadObservation, cumulativeReward?: number, episodeLength?: number, processedObs?: number[]) => void;
export type TerminatedAckHandler = (cumulativeReward: number, episodeLength: number, terminalReward: number, result: string) => void;
export type ConnectionHandler = (connected: boolean) => void;

export class AgentWebSocket {
    private ws: WebSocket | null = null;
    private _connected = false;
    private url: string;

    // Event handlers
    public onAction: ActionHandler | null = null;
    public onTerminatedAck: TerminatedAckHandler | null = null;
    public onConnectionChange: ConnectionHandler | null = null;

    constructor(url = 'ws://localhost:8765') {
        this.url = url;
    }

    get connected(): boolean {
        return this._connected;
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            console.log(`Connecting to agent server at ${this.url}...`);
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('Connected to agent server!');
                this._connected = true;
                this.onConnectionChange?.(true);
                resolve();
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };

            this.ws.onclose = () => {
                console.log('Disconnected from agent server');
                this._connected = false;
                this.onConnectionChange?.(false);
            };
        });
    }

    private handleMessage(event: MessageEvent): void {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'action') {
                this.onAction?.(
                    data.action,
                    data.action_name,
                    data.value,
                    data.observation,
                    data.cumulative_reward,
                    data.episode_length,
                    data.processed_obs
                );
            } else if (data.type === 'terminated_ack') {
                this.onTerminatedAck?.(
                    data.cumulative_reward,
                    data.episode_length,
                    data.terminal_reward,
                    data.result
                );
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    }

    sendReset(): void {
        if (this.ws && this._connected) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }
    }

    sendObservation(observation: JadObservation): void {
        if (this.ws && this._connected) {
            this.ws.send(JSON.stringify({ type: 'observation', observation }));
        }
    }

    sendTerminated(result: string, observation: JadObservation): void {
        if (this.ws && this._connected) {
            this.ws.send(JSON.stringify({ type: 'terminated', result, observation }));
        }
    }
}

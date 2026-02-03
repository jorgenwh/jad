import { StepResult } from '../core';

export type ActionHandler = (action: number[], value: number) => void;
export type ConnectionHandler = (connected: boolean) => void;

export class AgentWebSocket {
    private ws: WebSocket | null = null;
    private _connected = false;
    private url: string;

    public onAction: ActionHandler | null = null;
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
            this.onAction?.(data.action, data.value);
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    }

    sendReset(): void {
        if (this.ws && this._connected) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }
    }

    sendStep(step: StepResult): void {
        if (this.ws && this._connected) {
            this.ws.send(JSON.stringify({ type: 'step', ...step }));
        }
    }
}

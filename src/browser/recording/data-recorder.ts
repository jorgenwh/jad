import { Player } from 'osrs-sdk';
import { JadRegion } from '../../core/jad-region';
import { JadConfig, Observation, TerminationState } from '../../core/types';
import { buildObservation, countPotionDoses } from '../../core/observation';
import { checkTermination } from '../../core/utils';
import { ActionRecorder } from './action-recorder';

export interface RecordedStep {
    tick: number;
    observation: Observation;
    action: number[];  // MultiDiscrete: [protection, offensive, potion, target]
}

export interface RecordedEpisode {
    episodeId: string;
    config: JadConfig;
    startTime: string;
    endTime: string;
    outcome: 'kill' | 'death' | 'incomplete';
    totalTicks: number;
    steps: RecordedStep[];
}

export class DataRecorder {
    private player: Player;
    private jadRegion: JadRegion;
    private config: JadConfig;
    private actionRecorder: ActionRecorder;

    private currentEpisodeId: string;
    private steps: RecordedStep[] = [];
    private startTime: Date;
    private tickCount: number = 0;
    private isRecording: boolean = false;

    private startingDoses: { bastion: number; saraBrew: number; superRestore: number };

    constructor(
        player: Player,
        jadRegion: JadRegion,
        config: JadConfig,
        actionRecorder: ActionRecorder
    ) {
        this.player = player;
        this.jadRegion = jadRegion;
        this.config = config;
        this.actionRecorder = actionRecorder;

        this.currentEpisodeId = this.generateEpisodeId();
        this.startTime = new Date();
        this.startingDoses = this.captureStartingDoses();
    }

    startRecording(): void {
        this.isRecording = true;
        this.currentEpisodeId = this.generateEpisodeId();
        this.startTime = new Date();
        this.steps = [];
        this.tickCount = 0;
        this.startingDoses = this.captureStartingDoses();
        console.log(`[Recording] Started episode ${this.currentEpisodeId}`);
    }

    stopRecording(): void {
        this.isRecording = false;
        console.log(`[Recording] Stopped. ${this.steps.length} steps recorded.`);
    }

    tick(): TerminationState {
        if (!this.isRecording) {
            return TerminationState.ONGOING;
        }

        // Capture observation
        const observation = buildObservation(
            this.player,
            this.jadRegion,
            this.config,
            this.startingDoses
        );

        // Get action for this tick as MultiDiscrete array
        const action = this.actionRecorder.consumePendingAction();

        // Record step
        this.steps.push({
            tick: this.tickCount,
            observation,
            action,
        });

        this.tickCount++;

        // Check termination
        const termination = checkTermination(this.player, this.jadRegion, this.config);
        if (termination !== TerminationState.ONGOING) {
            this.onEpisodeEnd(termination);
        }

        return termination;
    }

    private onEpisodeEnd(termination: TerminationState): void {
        const outcome = termination === TerminationState.JAD_KILLED ? 'kill' : 'death';
        console.log(`[Recording] Episode ended: ${outcome} after ${this.tickCount} ticks`);

        const episode = this.buildEpisode(outcome);
        this.exportEpisode(episode);
        this.stopRecording();
    }

    private buildEpisode(outcome: 'kill' | 'death' | 'incomplete'): RecordedEpisode {
        return {
            episodeId: this.currentEpisodeId,
            config: { ...this.config },
            startTime: this.startTime.toISOString(),
            endTime: new Date().toISOString(),
            outcome,
            totalTicks: this.tickCount,
            steps: this.steps,
        };
    }

    private exportEpisode(episode: RecordedEpisode): void {
        const json = JSON.stringify(episode, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `jad_episode_${episode.episodeId}_${episode.outcome}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`[Recording] Exported ${episode.steps.length} steps to ${a.download}`);
    }

    exportCurrentData(): void {
        if (this.steps.length === 0) {
            console.log('[Recording] No data to export');
            return;
        }

        const episode = this.buildEpisode('incomplete');
        this.exportEpisode(episode);
    }

    getStats(): { tickCount: number; stepCount: number; isRecording: boolean } {
        return {
            tickCount: this.tickCount,
            stepCount: this.steps.length,
            isRecording: this.isRecording,
        };
    }

    private generateEpisodeId(): string {
        const now = new Date();
        const timestamp = now.toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
        const random = Math.random().toString(36).substring(2, 6);
        return `${timestamp}_${random}`;
    }

    private captureStartingDoses(): { bastion: number; saraBrew: number; superRestore: number } {
        const doses = countPotionDoses(this.player);
        return {
            bastion: doses.bastionDoses,
            saraBrew: doses.saraBrewDoses,
            superRestore: doses.superRestoreDoses,
        };
    }
}

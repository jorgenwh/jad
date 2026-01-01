import { Region, Player, UnitTypes } from 'osrs-sdk';
import { InvisibleMovementBlocker } from 'osrs-sdk';

import { getRangedLoadout } from './loadout';
import { Jad } from './jad';
import { YtHurKot } from './healer';
import { JadConfig, DEFAULT_CONFIG, HealerAggro } from './types';

const REGION_WIDTH = 27;
const REGION_HEIGHT = 27;
const PLAYER_SPAWN = { x: 13, y: 13 };

const SINGLE_JAD_ATTACK_SPEED = 8;
const MULTI_JAD_ATTACK_SPEED = 9;

const JAD_SPAWN_POSITIONS: { x: number; y: number }[][] = [
    // 1 Jad
    [
        { x: 11, y: 22 },
    ],
    // 2 Jads
    [
        { x: 7, y: 22 },
        { x: 15, y: 22 },
    ],
    // 3 Jads
    [
        { x: 11, y: 22 },
        { x: 18, y: 15 },
        { x: 4, y: 15 },
    ],
    // 4 Jads
    [
        { x: 11, y: 22 },
        { x: 18, y: 15 },
        { x: 4, y: 15 },
        { x: 11, y: 8 },
    ],
    // 5 Jads
    [
        { x: 7, y: 22 },
        { x: 15, y: 22 },
        { x: 18, y: 15 },
        { x: 4, y: 15 },
        { x: 11, y: 8 },
    ],
    // 6 Jads
    [
        { x: 7, y: 22 },
        { x: 15, y: 22 },
        { x: 18, y: 15 },
        { x: 4, y: 15 },
        { x: 7, y: 8 },
        { x: 15, y: 8 },
    ],
];

function shuffleArray<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

export class JadRegion extends Region {
    private config: JadConfig;
    private _jads: Jad[] = [];
    private _healers: Map<number, YtHurKot[]> = new Map();

    constructor(config: JadConfig = DEFAULT_CONFIG) {
        super();
        this.config = config;
    }

    getName(): string {
        return 'Jad';
    }

    get width(): number {
        return REGION_WIDTH;
    }

    get height(): number {
        return REGION_HEIGHT;
    }

    getJad(index: number): Jad | null {
        if (index < 0 || index >= this._jads.length) {
            return null;
        }

        const jad = this._jads[index];
        if (!jad) {
            console.warn(`Jad ${index} is null`);
            return null;
        }

        // Check both isDying and HP to handle timing gap between HP=0 and death animation
        if (jad.isDying() || jad.currentStats.hitpoint <= 0) {
            return null;
        }

        return jad;
    }

    registerHealers(jadIndex: number, healers: YtHurKot[]): void {
        if (jadIndex < 0 || jadIndex >= this.config.jadCount) {
            return;
        }
        this._healers.set(jadIndex, healers);
    }

    getHealer(jadIndex: number, healerIndex: number): YtHurKot | null {
        const jad = this._jads[jadIndex];
        if (jad && (jad.isDying() || jad.currentStats.hitpoint <= 0)) {
            return null;
        }

        const healers = this._healers.get(jadIndex);
        if (!healers) {
            return null;
        }

        if (healerIndex < 0 || healerIndex >= healers.length) {
            return null;
        }

        const healer = healers[healerIndex];
        if (!healer) {
            return null;
        }

        // Check both isDying and HP to handle timing gap between HP=0 and death animation
        if (healer.isDying() || healer.currentStats.hitpoint <= 0) {
            return null;
        }

        return healer;
    }

    getHealerAggro(jadIndex: number, healerIndex: number): HealerAggro {
        const healer = this.getHealer(jadIndex, healerIndex);
        if (!healer) {
            return HealerAggro.NOT_PRESENT;
        }

        const aggro = healer.aggro;
        if (aggro && aggro.type === UnitTypes.PLAYER) {
            return HealerAggro.PLAYER;
        }
        return HealerAggro.JAD;
    }

    initialiseRegion(): { player: Player } {
        const player = new Player(this, PLAYER_SPAWN);
        this.addPlayer(player);

        player.setUnitOptions(getRangedLoadout());
        this.addBoundaryBlockers();

        const spawnPositions = JAD_SPAWN_POSITIONS[this.config.jadCount - 1] || JAD_SPAWN_POSITIONS[0];
        const attackSpeed = this.config.jadCount === 1 ? SINGLE_JAD_ATTACK_SPEED : MULTI_JAD_ATTACK_SPEED;

        const attackOrder = shuffleArray(
            Array.from({ length: this.config.jadCount }, (_, i) => i)
        );

        // Spawn Jads
        for (let i = 0; i < this.config.jadCount; i++) {
            const pos = spawnPositions[i];
            // Use shuffled position in attack order for this Jad's timing
            const attackOffset = 1 + Math.floor((attackOrder[i] * attackSpeed) / this.config.jadCount);

            const jad = new Jad(this, pos, i, this.config.healersPerJad, attackSpeed, {
                aggro: player,
                cooldown: attackOffset,  // Initial attackDelay offset
            });
            this._jads.push(jad);
            this.addMob(jad);
        }

        return { player };
    }

    addBoundaryBlockers() {
        // Top and bottom edges
        for (let x = -1; x <= this.width; x++) {
            this.addEntity(new InvisibleMovementBlocker(this, { x, y: -1 }));
            this.addEntity(new InvisibleMovementBlocker(this, { x, y: this.height }));
        }
        // Left and right edges
        for (let y = 0; y < this.height; y++) {
            this.addEntity(new InvisibleMovementBlocker(this, { x: -1, y }));
            this.addEntity(new InvisibleMovementBlocker(this, { x: this.width, y }));
        }
    }

    postTick() {
        super.postTick();
        this.updateTickCounter();
    }

    private updateTickCounter() {
        const tickCountElement = document.getElementById('tick_count');
        if (tickCountElement && this.world) {
            tickCountElement.innerText = String(this.world.globalTickCounter);
        }
    }

    drawWorldBackground(context: OffscreenCanvasRenderingContext2D, scale: number) {
        // Draw a checkerboard pattern for the floor
        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                // Alternate colors for checkerboard
                if ((x + y) % 2 === 0) {
                    context.fillStyle = '#3d3d29';  // Dark tan
                } else {
                    context.fillStyle = '#4a4a35';  // Light tan
                }
                context.fillRect(x * scale, y * scale, scale, scale);
            }
        }

        // Draw grid lines
        context.strokeStyle = '#2a2a1a';
        context.lineWidth = 1;
        for (let x = 0; x <= this.width; x++) {
            context.beginPath();
            context.moveTo(x * scale, 0);
            context.lineTo(x * scale, this.height * scale);
            context.stroke();
        }
        for (let y = 0; y <= this.height; y++) {
            context.beginPath();
            context.moveTo(0, y * scale);
            context.lineTo(this.width * scale, y * scale);
            context.stroke();
        }
    }
}

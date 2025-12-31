import { Region, Player, UnitTypes } from 'osrs-sdk';
import { InvisibleMovementBlocker } from 'osrs-sdk';

import { getRangedLoadout } from './loadout';
import { Jad } from './jad';
import { YtHurKot } from './healer';
import { JadConfig, DEFAULT_CONFIG } from './types';

// Healer aggro state for observation
export enum HealerAggro {
    NOT_PRESENT = 0,
    JAD = 1,
    PLAYER = 2,
}

// Region dimensions - sized for 6 Jads in a circle around player
const REGION_WIDTH = 27;
const REGION_HEIGHT = 27;
const PLAYER_SPAWN = { x: 13, y: 13 };  // Center of region

// Attack speeds: 8 ticks for single Jad, 9 ticks for multi-Jad (like InfernoTrainer)
const SINGLE_JAD_ATTACK_SPEED = 8;
const MULTI_JAD_ATTACK_SPEED = 9;

// Hardcoded spawn positions for 1-6 Jads (relative to 27x27 region with player at 13,13)
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

/**
 * Get attack speed based on Jad count.
 * Single Jad uses 8 ticks, multi-Jad uses 9 ticks (like InfernoTrainer).
 */
function getAttackSpeed(jadCount: number): number {
    return jadCount === 1 ? SINGLE_JAD_ATTACK_SPEED : MULTI_JAD_ATTACK_SPEED;
}

/**
 * Calculate attack offset for a given position in the attack order.
 * Evenly distributes first attacks across the attack cycle.
 * @param orderPosition - Position in the attack order (0 = attacks first)
 * @param jadCount - Total number of Jads
 */
function getAttackOffset(orderPosition: number, jadCount: number): number {
    const attackSpeed = getAttackSpeed(jadCount);
    // Stagger attacks evenly across the attack speed window
    // E.g., for 3 Jads with attack speed 9: offsets of 1, 4, 7 (like InfernoTrainer)
    return 1 + Math.floor((orderPosition * attackSpeed) / jadCount);
}

/**
 * Fisher-Yates shuffle for randomizing attack order.
 */
function shuffleArray<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Healer array type (3 healers per Jad)
type HealerTuple = [YtHurKot | null, YtHurKot | null, YtHurKot | null];

export class JadRegion extends Region {
    private config: JadConfig;
    private _jads: Jad[] = [];

    // Per-Jad healer tracking: Map<jadIndex, [healer0, healer1, healer2]>
    private _healers: Map<number, HealerTuple> = new Map();
    private _healersSpawnedPerJad: Map<number, boolean> = new Map();

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

    get jadCount(): number {
        return this.config.jadCount;
    }

    get healersPerJad(): number {
        return this.config.healersPerJad;
    }

    /**
     * Get Jad by index (0-based).
     */
    getJad(index: number): Jad | null {
        if (index < 0 || index >= this._jads.length) return null;
        const jad = this._jads[index];
        // Check both isDying and HP to handle timing gap between HP=0 and death animation
        if (jad && (jad.isDying() || jad.currentStats.hitpoint <= 0)) return null;
        return jad;
    }

    /**
     * Get the first (primary) Jad - for backwards compatibility.
     */
    get jad(): Jad {
        if (this._jads.length === 0) {
            throw new Error('Jad not initialized');
        }
        return this._jads[0];
    }

    /**
     * Get all alive Jads.
     */
    get jads(): Jad[] {
        return this._jads.filter((j) => !j.isDying());
    }

    /**
     * Register a healer at a specific Jad and healer index.
     * Called by Jad when spawning healers to maintain stable indices.
     */
    registerHealer(jadIndex: number, healerIndex: number, healer: YtHurKot): void {
        if (jadIndex < 0 || jadIndex >= this.config.jadCount) return;
        if (healerIndex < 0 || healerIndex >= this.config.healersPerJad) return;

        let healerTuple = this._healers.get(jadIndex);
        if (!healerTuple) {
            healerTuple = [null, null, null];
            this._healers.set(jadIndex, healerTuple);
        }
        healerTuple[healerIndex] = healer;
        this._healersSpawnedPerJad.set(jadIndex, true);
    }

    /**
     * Get healer at specific Jad and healer index.
     * Returns null if not spawned, dead, or parent Jad is dead.
     */
    getHealer(jadIndex: number, healerIndex: number): YtHurKot | null {
        // If the parent Jad is dead, healers despawn
        const jad = this._jads[jadIndex];
        if (jad && (jad.isDying() || jad.currentStats.hitpoint <= 0)) return null;

        const healerTuple = this._healers.get(jadIndex);
        if (!healerTuple) return null;
        if (healerIndex < 0 || healerIndex >= 3) return null;

        const healer = healerTuple[healerIndex];
        // Check both isDying and HP to handle timing gap between HP=0 and death animation
        if (healer && (healer.isDying() || healer.currentStats.hitpoint <= 0)) return null;
        return healer;
    }

    /**
     * Get healer aggro state for observation.
     */
    getHealerAggro(jadIndex: number, healerIndex: number): HealerAggro {
        const healer = this.getHealer(jadIndex, healerIndex);
        if (!healer) return HealerAggro.NOT_PRESENT;

        const aggro = healer.aggro;
        if (aggro && aggro.type === UnitTypes.PLAYER) {
            return HealerAggro.PLAYER;
        }
        return HealerAggro.JAD;
    }

    /**
     * Whether healers have spawned for a specific Jad.
     */
    hasHealersSpawned(jadIndex: number): boolean {
        return this._healersSpawnedPerJad.get(jadIndex) ?? false;
    }

    /**
     * Whether any healers have spawned this episode.
     */
    get healersSpawned(): boolean {
        for (const [, spawned] of this._healersSpawnedPerJad) {
            if (spawned) return true;
        }
        return false;
    }

    /**
     * Get all alive healers in the region (for backwards compatibility).
     * Excludes healers whose Jads are dead.
     */
    get healers(): YtHurKot[] {
        const result: YtHurKot[] = [];
        for (const [jadIndex, healerTuple] of this._healers) {
            // Skip healers if their Jad is dead
            const jad = this._jads[jadIndex];
            if (jad && (jad.isDying() || jad.currentStats.hitpoint <= 0)) continue;

            for (const healer of healerTuple) {
                if (healer && !healer.isDying() && healer.currentStats.hitpoint > 0) {
                    result.push(healer);
                }
            }
        }
        return result;
    }

    /**
     * Get the count of alive healers.
     */
    get healerCount(): number {
        return this.healers.length;
    }

    initialiseRegion(): { player: Player } {
        // Spawn player in center of region
        const player = new Player(this, PLAYER_SPAWN);
        this.addPlayer(player);

        // Use ranged loadout
        player.setUnitOptions(getRangedLoadout());

        this.addBoundaryBlockers();

        // Spawn Jads around the player with staggered attack timing
        const spawnPositions = JAD_SPAWN_POSITIONS[this.config.jadCount - 1] || JAD_SPAWN_POSITIONS[0];
        const attackSpeed = getAttackSpeed(this.config.jadCount);

        // Randomize attack order each episode to prevent learning bias
        // where the agent always targets a specific Jad based on attack timing
        const attackOrder = shuffleArray(
            Array.from({ length: this.config.jadCount }, (_, i) => i)
        );

        for (let i = 0; i < this.config.jadCount; i++) {
            const pos = spawnPositions[i];
            // Use shuffled position in attack order for this Jad's timing
            const attackOffset = getAttackOffset(attackOrder[i], this.config.jadCount);
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

import { Region, Player, Unit, UnitTypes } from 'osrs-sdk';
import { InvisibleMovementBlocker } from 'osrs-sdk';

import { getRangedLoadout, getMeleeLoadout } from './loadout';
import { Jad } from './jad';
import { YtHurKot } from './healer';

// Healer aggro state for observation
export enum HealerAggro {
  NOT_PRESENT = 0,
  JAD = 1,
  PLAYER = 2,
}

export class JadRegion extends Region {
  getName(): string {
    return 'Jad';
  }

  get width(): number {
    return 20;
  }

  get height(): number {
    return 20;
  }

  private _jad: Jad | null = null;

  // Fixed-size healer tracking array - indices are stable for the episode
  private _healers: [YtHurKot | null, YtHurKot | null, YtHurKot | null] = [null, null, null];
  private _healersSpawned = false;

  get jad(): Jad {
    if (!this._jad) {
      throw new Error('Jad not initialized');
    }
    return this._jad;
  }

  /**
   * Register a healer at a specific index (0, 1, or 2).
   * Called by Jad when spawning healers to maintain stable indices.
   */
  registerHealer(index: number, healer: YtHurKot): void {
    if (index >= 0 && index < 3) {
      this._healers[index] = healer;
      this._healersSpawned = true;
    }
  }

  /**
   * Get healer at specific index (0, 1, or 2).
   * Returns null if not spawned or dead.
   */
  getHealer(index: number): YtHurKot | null {
    if (index < 0 || index >= 3) return null;
    const healer = this._healers[index];
    // Return null if healer is dead/dying
    if (healer && healer.isDying()) return null;
    return healer;
  }

  /**
   * Get healer aggro state for observation.
   */
  getHealerAggro(index: number): HealerAggro {
    const healer = this._healers[index];
    if (!healer || healer.isDying()) return HealerAggro.NOT_PRESENT;

    const aggro = healer.aggro;
    if (aggro && aggro.type === UnitTypes.PLAYER) {
      return HealerAggro.PLAYER;
    }
    return HealerAggro.JAD;
  }

  /**
   * Whether healers have spawned this episode.
   */
  get healersSpawned(): boolean {
    return this._healersSpawned;
  }

  /**
   * Get all alive healers in the region (for backwards compatibility).
   */
  get healers(): YtHurKot[] {
    return this._healers.filter((h): h is YtHurKot => h !== null && !h.isDying());
  }

  /**
   * Get the count of alive healers.
   */
  get healerCount(): number {
    return this.healers.length;
  }

  initialiseRegion(): { player: Player } {
    const player = new Player(this, { x: 10, y: 5 });
    this.addPlayer(player);

    player.setUnitOptions(getMeleeLoadout());

    this.addBoundaryBlockers();

    this._jad = new Jad(this, { x: 8, y: 15 }, { aggro: player });
    this.addMob(this._jad);

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

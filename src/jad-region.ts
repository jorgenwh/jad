import { Region, Player } from 'osrs-sdk';
import { InvisibleMovementBlocker } from 'osrs-sdk';

import { getRangedLoadout, getMeleeLoadout } from './loadout';
import { Jad } from './jad';
import { YtHurKot } from './healer';

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

  get jad(): Jad {
    if (!this._jad) {
      throw new Error('Jad not initialized');
    }
    return this._jad;
  }

  /**
   * Get all alive healers in the region.
   */
  get healers(): YtHurKot[] {
    return this.mobs.filter((mob): mob is YtHurKot => mob instanceof YtHurKot && !mob.isDying());
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

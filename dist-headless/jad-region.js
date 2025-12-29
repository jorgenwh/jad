"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.JadRegion = void 0;
const osrs_sdk_1 = require("osrs-sdk");
const osrs_sdk_2 = require("osrs-sdk");
const loadout_1 = require("./loadout");
const jad_1 = require("./jad");
class JadRegion extends osrs_sdk_1.Region {
    constructor() {
        super(...arguments);
        this._jad = null;
    }
    getName() {
        return 'Jad';
    }
    get width() {
        return 20;
    }
    get height() {
        return 20;
    }
    get jad() {
        if (!this._jad) {
            throw new Error('Jad not initialized');
        }
        return this._jad;
    }
    initialiseRegion() {
        const player = new osrs_sdk_1.Player(this, { x: 10, y: 5 });
        this.addPlayer(player);
        player.setUnitOptions((0, loadout_1.getMeleeLoadout)());
        this.addBoundaryBlockers();
        this._jad = new jad_1.Jad(this, { x: 8, y: 15 }, { aggro: player });
        this.addMob(this._jad);
        return { player };
    }
    addBoundaryBlockers() {
        // Top and bottom edges
        for (let x = -1; x <= this.width; x++) {
            this.addEntity(new osrs_sdk_2.InvisibleMovementBlocker(this, { x, y: -1 }));
            this.addEntity(new osrs_sdk_2.InvisibleMovementBlocker(this, { x, y: this.height }));
        }
        // Left and right edges
        for (let y = 0; y < this.height; y++) {
            this.addEntity(new osrs_sdk_2.InvisibleMovementBlocker(this, { x: -1, y }));
            this.addEntity(new osrs_sdk_2.InvisibleMovementBlocker(this, { x: this.width, y }));
        }
    }
    drawWorldBackground(context, scale) {
        // Draw a checkerboard pattern for the floor
        for (let x = 0; x < this.width; x++) {
            for (let y = 0; y < this.height; y++) {
                // Alternate colors for checkerboard
                if ((x + y) % 2 === 0) {
                    context.fillStyle = '#3d3d29'; // Dark tan
                }
                else {
                    context.fillStyle = '#4a4a35'; // Light tan
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
exports.JadRegion = JadRegion;

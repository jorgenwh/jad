"use strict";
/**
 * Headless environment wrapper for RL training.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.HeadlessEnv = exports.JadAction = void 0;
const osrs_sdk_1 = require("osrs-sdk");
// Initialize settings from storage (uses mock localStorage)
osrs_sdk_1.Settings.readFromStorage();
// Action enum
var JadAction;
(function (JadAction) {
    JadAction[JadAction["WAIT"] = 0] = "WAIT";
    JadAction[JadAction["PRAY_MAGE"] = 1] = "PRAY_MAGE";
    JadAction[JadAction["PRAY_RANGE"] = 2] = "PRAY_RANGE";
    JadAction[JadAction["DRINK_RESTORE"] = 3] = "DRINK_RESTORE";
    JadAction[JadAction["ATTACK"] = 4] = "ATTACK";
})(JadAction || (exports.JadAction = JadAction = {}));
class HeadlessEnv {
    constructor(createRegion) {
        // Track Jad's attack for observation
        this.currentJadAttack = 0; // 0=none, 1=mage, 2=range
        this.attackTicksRemaining = 0;
        this.prevJadAttackDelay = 0; // To detect when Jad actually attacks
        this.createRegion = createRegion;
        this.world = new osrs_sdk_1.World();
        this.region = createRegion();
        this.initializeRegion();
    }
    initializeRegion() {
        this.region.world = this.world;
        this.world.addRegion(this.region);
        // Initialize region and get player
        const result = this.region.initialiseRegion();
        this.player = result.player;
        // Get Jad from the region's jad getter
        this.jad = this.region.jad;
        // Set player in Trainer (used by osrs-sdk internals)
        osrs_sdk_1.Trainer.setPlayer(this.player);
        // Initialize attack tracking
        this.currentJadAttack = 0;
        this.attackTicksRemaining = 0;
        this.prevJadAttackDelay = this.jad.attackDelay;
    }
    reset() {
        // Clear and recreate
        this.world = new osrs_sdk_1.World();
        this.region = this.createRegion();
        this.initializeRegion();
        return {
            observation: this.getObservation(),
        };
    }
    step(action) {
        // Execute action before tick
        this.executeAction(action);
        // Observe Jad's attack style before tick (for tracking)
        this.updateJadAttackTracking();
        // Advance simulation by one tick
        this.world.tickWorld(1);
        // Get observation after tick
        const observation = this.getObservation();
        // Check termination
        const playerDead = this.player.dying > 0 || this.player.currentStats.hitpoint <= 0;
        const jadDead = this.jad.dying > 0 || this.jad.currentStats.hitpoint <= 0;
        const terminated = playerDead || jadDead;
        return {
            observation,
            terminated,
        };
    }
    executeAction(action) {
        switch (action) {
            case JadAction.WAIT:
                // Do nothing
                break;
            case JadAction.PRAY_MAGE:
                this.setPrayer('magic');
                break;
            case JadAction.PRAY_RANGE:
                this.setPrayer('range');
                break;
            case JadAction.DRINK_RESTORE:
                this.drinkRestore();
                break;
            case JadAction.ATTACK:
                this.attackJad();
                break;
        }
    }
    setPrayer(style) {
        const prayerController = this.player.prayerController;
        if (!prayerController)
            return;
        // Map style to prayer name
        const prayerName = style === 'magic' ? 'Protect from Magic' : 'Protect from Range';
        // Find the prayer by name (in all prayers, not just active ones)
        const targetPrayer = prayerController.findPrayerByName(prayerName);
        if (targetPrayer && !targetPrayer.isActive) {
            // Activate the prayer - this auto-deactivates conflicting overhead prayers
            targetPrayer.activate(this.player);
        }
    }
    // TODO: find a better way to identify restore potions
    drinkRestore() {
        // Find a super restore in inventory and drink it
        if (!this.player || !this.player.inventory)
            return;
        const inventory = this.player.inventory;
        for (const item of inventory) {
            if (item && item instanceof osrs_sdk_1.Potion && item.doses > 0) {
                // Check if it's a restore-type potion by checking itemName
                const itemName = item.itemName?.toString().toLowerCase() || '';
                if (itemName.includes('restore')) {
                    item.drink(this.player);
                    break;
                }
            }
        }
    }
    attackJad() {
        // Set Jad as aggro target
        this.player.setAggro(this.jad);
    }
    updateJadAttackTracking() {
        // Track Jad's attack style
        // The attack style is visible when Jad attacks and persists until damage lands
        const currentDelay = this.jad.attackDelay;
        // Detect actual attack: attackDelay was low (0-1), now it's high (just reset after attacking)
        if (this.prevJadAttackDelay <= 1 && currentDelay > 1) {
            // Jad just attacked - capture the attack style NOW
            const style = this.jad.attackStyle;
            this.currentJadAttack = style === 'magic' ? 1 : 2;
            this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
        }
        // Decrement visibility timer
        if (this.attackTicksRemaining > 0) {
            this.attackTicksRemaining--;
            if (this.attackTicksRemaining === 0) {
                this.currentJadAttack = 0;
            }
        }
        this.prevJadAttackDelay = currentDelay;
    }
    getObservation() {
        // Get active prayer (0=none, 1=mage, 2=range)
        let activePrayer = 0;
        const prayerController = this.player.prayerController;
        if (prayerController) {
            // Check which overhead protection prayer is active by name
            const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
            const rangePrayer = prayerController.findPrayerByName('Protect from Range');
            if (magicPrayer?.isActive) {
                activePrayer = 1;
            }
            else if (rangePrayer?.isActive) {
                activePrayer = 2;
            }
        }
        // Count restore doses
        let restoreDoses = 0;
        if (this.player && this.player.inventory) {
            const inventory = this.player.inventory;
            for (const item of inventory) {
                if (item && item instanceof osrs_sdk_1.Potion && item.doses > 0) {
                    const itemName = item.itemName?.toString().toLowerCase() || '';
                    if (itemName.includes('restore')) {
                        restoreDoses += item.doses;
                    }
                }
            }
        }
        return {
            player_hp: this.player?.currentStats?.hitpoint ?? 0,
            player_prayer: this.player?.currentStats?.prayer ?? 0,
            active_prayer: activePrayer,
            jad_hp: this.jad?.currentStats?.hitpoint ?? 0,
            jad_attack: this.currentJadAttack,
            restore_doses: restoreDoses,
        };
    }
}
exports.HeadlessEnv = HeadlessEnv;

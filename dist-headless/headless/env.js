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
    JadAction[JadAction["PRAY_MELEE"] = 5] = "PRAY_MELEE";
    JadAction[JadAction["DRINK_SUPER_COMBAT"] = 6] = "DRINK_SUPER_COMBAT";
    JadAction[JadAction["TOGGLE_PIETY"] = 7] = "TOGGLE_PIETY";
    JadAction[JadAction["DRINK_SARA_BREW"] = 8] = "DRINK_SARA_BREW";
})(JadAction || (exports.JadAction = JadAction = {}));
class HeadlessEnv {
    constructor(createRegion) {
        // Track Jad's attack for observation
        this.currentJadAttack = 0; // 0=none, 1=mage, 2=range, 3=melee
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
                this.togglePrayer('Protect from Magic');
                break;
            case JadAction.PRAY_RANGE:
                this.togglePrayer('Protect from Range');
                break;
            case JadAction.DRINK_RESTORE:
                this.drinkPotion('restore');
                break;
            case JadAction.ATTACK:
                this.attackJad();
                break;
            case JadAction.PRAY_MELEE:
                this.togglePrayer('Protect from Melee');
                break;
            case JadAction.DRINK_SUPER_COMBAT:
                this.drinkPotion('super combat');
                break;
            case JadAction.TOGGLE_PIETY:
                this.togglePrayer('Piety');
                break;
            case JadAction.DRINK_SARA_BREW:
                this.drinkPotion('saradomin brew');
                break;
        }
    }
    togglePrayer(prayerName) {
        const prayerController = this.player.prayerController;
        if (!prayerController)
            return;
        // Find the prayer by name
        const targetPrayer = prayerController.findPrayerByName(prayerName);
        if (targetPrayer) {
            if (targetPrayer.isActive) {
                // Deactivate if already active
                targetPrayer.deactivate();
            }
            else {
                // Activate if not active - this auto-deactivates conflicting overhead prayers
                targetPrayer.activate(this.player);
            }
        }
    }
    drinkPotion(potionType) {
        // Find a potion in inventory and drink it
        if (!this.player || !this.player.inventory)
            return;
        const inventory = this.player.inventory;
        for (const item of inventory) {
            if (item && item instanceof osrs_sdk_1.Potion && item.doses > 0) {
                const itemName = item.itemName?.toString().toLowerCase() || '';
                if (itemName.includes(potionType)) {
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
            if (style === 'magic') {
                this.currentJadAttack = 1;
                this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
            }
            else if (style === 'range') {
                this.currentJadAttack = 2;
                this.attackTicksRemaining = 4; // Visible for 4 ticks (attack + 3 projectile flight)
            }
            else {
                // Melee attack - style is 'stab' (from canMeleeIfClose)
                // Melee is instant, no projectile delay
                // Use 2 so after immediate decrement it's 1 (visible for 1 tick)
                this.currentJadAttack = 3;
                this.attackTicksRemaining = 2;
            }
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
        // Get active prayer (0=none, 1=mage, 2=range, 3=melee)
        let activePrayer = 0;
        let pietyActive = false;
        const prayerController = this.player.prayerController;
        if (prayerController) {
            // Check which overhead protection prayer is active by name
            const magicPrayer = prayerController.findPrayerByName('Protect from Magic');
            const rangePrayer = prayerController.findPrayerByName('Protect from Range');
            const meleePrayer = prayerController.findPrayerByName('Protect from Melee');
            const pietyPrayer = prayerController.findPrayerByName('Piety');
            if (magicPrayer?.isActive) {
                activePrayer = 1;
            }
            else if (rangePrayer?.isActive) {
                activePrayer = 2;
            }
            else if (meleePrayer?.isActive) {
                activePrayer = 3;
            }
            pietyActive = pietyPrayer?.isActive ?? false;
        }
        // Count potion doses
        let restoreDoses = 0;
        let superCombatDoses = 0;
        let saraBrewDoses = 0;
        if (this.player && this.player.inventory) {
            const inventory = this.player.inventory;
            for (const item of inventory) {
                if (item && item instanceof osrs_sdk_1.Potion && item.doses > 0) {
                    const itemName = item.itemName?.toString().toLowerCase() || '';
                    if (itemName.includes('restore')) {
                        restoreDoses += item.doses;
                    }
                    else if (itemName.includes('super combat')) {
                        superCombatDoses += item.doses;
                    }
                    else if (itemName.includes('saradomin brew')) {
                        saraBrewDoses += item.doses;
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
            super_combat_doses: superCombatDoses,
            sara_brew_doses: saraBrewDoses,
            piety_active: pietyActive,
        };
    }
}
exports.HeadlessEnv = HeadlessEnv;
